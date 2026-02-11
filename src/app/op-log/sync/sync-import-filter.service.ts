import { inject, Injectable } from '@angular/core';
import { OperationLogStoreService } from '../persistence/operation-log-store.service';
import { Operation, OpType } from '../core/operation.types';
import {
  compareVectorClocks,
  limitVectorClockSize,
  VectorClock,
  VectorClockComparison,
  vectorClockToString,
} from '../../core/util/vector-clock';
import { MAX_VECTOR_CLOCK_SIZE } from '../core/operation-log.const';
import { OpLog } from '../../core/log';

/**
 * Service responsible for filtering operations invalidated by SYNC_IMPORT, BACKUP_IMPORT, or REPAIR operations.
 *
 * ## The Problem
 * ```
 * Timeline:
 *   Client A creates ops → Client B does SYNC_IMPORT → Client A syncs
 *
 * Result:
 *   - Client A's ops have higher serverSeq than SYNC_IMPORT
 *   - But they reference entities that were WIPED by the import
 *   - Applying them causes "Task not found" errors
 * ```
 *
 * ## The Solution: Clean Slate Semantics
 * SYNC_IMPORT and BACKUP_IMPORT are explicit user actions to restore ALL clients
 * to a specific state. ALL operations without knowledge of the import are dropped:
 *
 * - **GREATER_THAN / EQUAL**: Op was created with knowledge of import → KEEP
 * - **CONCURRENT**: Op was created without knowledge of import → DROP
 * - **LESS_THAN**: Op is dominated by import → DROP
 *
 * This ensures a true "restore to point in time" semantic. Concurrent work from
 * other clients is intentionally discarded because the user explicitly chose to
 * reset all state to the imported snapshot.
 *
 * We use vector clock comparison (not UUIDv7 timestamps) because vector clocks
 * track CAUSALITY (did the client know about the import?) rather than wall-clock
 * time (which can be affected by clock drift).
 */
/**
 * Detects whether a CONCURRENT comparison result is a pruning artifact rather
 * than genuine concurrency (import-side pruning).
 *
 * When a new client joins after a SYNC_IMPORT that already has MAX_VECTOR_CLOCK_SIZE
 * entries, the new client's ops get an (MAX+1)-entry clock. The server prunes this
 * back to MAX, dropping one inherited entry. When other clients compare this pruned
 * clock with the import's clock, both have MAX entries with different unique keys,
 * causing compareVectorClocks to return CONCURRENT even though the op was created
 * AFTER the import with full causal knowledge.
 *
 * Detection criteria (all must be true):
 * 1. Op's clientId is NOT in import's clock → client was born after the import
 * 2. Import clock has >= MAX_VECTOR_CLOCK_SIZE entries (pruning only happens at MAX)
 * 3. There are shared keys between the clocks
 * 4. ALL shared keys have op values >= import values → client inherited import's knowledge
 *
 * Known limitation: A false positive is theoretically possible if the import clock
 * itself was pruned and a genuinely concurrent client's ID happened to be among the
 * pruned entries. In that scenario the client would appear "born after" the import
 * (criterion 1) even though it existed before. This is unlikely in practice because
 * it requires the concurrent client to be one of the oldest (least-recently-updated)
 * entries in the import clock at the time of pruning.
 */
export const isLikelyPruningArtifact = (
  opClock: VectorClock,
  opClientId: string,
  importClock: VectorClock,
): boolean => {
  // If the op's clientId exists in the import's clock, the client existed before
  // the import. CONCURRENT is genuine - the client created ops without seeing it.
  if (opClientId in importClock) {
    return false;
  }

  // Server-side pruning (limitVectorClockSize) only drops entries when the clock
  // exceeds MAX_VECTOR_CLOCK_SIZE. If the import has fewer entries, the server
  // wouldn't have pruned the new client's clock, so CONCURRENT is genuine.
  const importKeyCount = Object.keys(importClock).length;
  if (importKeyCount < MAX_VECTOR_CLOCK_SIZE) {
    return false;
  }

  // Find shared keys between op and import clocks
  const opKeys = Object.keys(opClock);
  const sharedKeys = opKeys.filter((k) => k in importClock);

  // Need shared keys to make this determination. If there are none,
  // the op has no evidence of having seen the import.
  if (sharedKeys.length === 0) {
    return false;
  }

  // All shared keys must have op values >= import values.
  // If any shared key has op < import, the client has LESS knowledge
  // than the import for that client, meaning genuine concurrency.
  for (const key of sharedKeys) {
    if (opClock[key] < importClock[key]) {
      return false;
    }
  }

  return true;
};

/**
 * Detects whether a CONCURRENT comparison result is caused by server-side pruning
 * of the OP's vector clock, which dropped the import's client ID (op-side pruning).
 *
 * This is the inverse of isLikelyPruningArtifact:
 * - isLikelyPruningArtifact: import clock at MAX, new client's ID pruned from it
 * - isOpSidePruningArtifact: op clock at MAX, import's client ID pruned from it
 *
 * This scenario occurs when:
 * 1. A client creates a SYNC_IMPORT with a small vector clock (e.g., fresh client: {clientId:1})
 * 2. Other clients apply the import, merging {clientId:1} into their existing MAX-size clocks
 * 3. When creating new ops, the server prunes the import client's low-counter entry
 *    (it has value 1, the lowest in the clock, so it's pruned first)
 * 4. The resulting op's clock has no knowledge of the import client, causing CONCURRENT
 *
 * Detection criteria (all must be true):
 * 1. Op's clock has >= MAX_VECTOR_CLOCK_SIZE entries (server pruning occurred on op)
 * 2. Import's clock has < MAX_VECTOR_CLOCK_SIZE entries (small import, e.g., from fresh client)
 * 3. Import's client ID is NOT in op's clock (was pruned due to low counter value)
 * 4. One of:
 *    a. Shared keys exist and ALL have op values >= import values (op inherited import knowledge)
 *    b. No shared keys but op was created after the import (UUIDv7 ordering)
 *
 * @param opClock The op's vector clock
 * @param importClock The SYNC_IMPORT's vector clock (possibly normalized)
 * @param importClientId The client ID that created the SYNC_IMPORT
 * @param opId The op's UUIDv7 ID (used for timestamp comparison when clocks are disjoint)
 * @param importId The SYNC_IMPORT's UUIDv7 ID (used for timestamp comparison)
 */
export const isOpSidePruningArtifact = (
  opClock: VectorClock,
  importClock: VectorClock,
  importClientId: string,
  opId: string,
  importId: string,
): boolean => {
  const opKeyCount = Object.keys(opClock).length;
  const importKeyCount = Object.keys(importClock).length;

  // Op must be at MAX size (server pruning must have occurred)
  if (opKeyCount < MAX_VECTOR_CLOCK_SIZE) {
    return false;
  }

  // Import must be small (from a fresh/new client).
  // If import is at MAX, the existing isLikelyPruningArtifact handles it.
  if (importKeyCount >= MAX_VECTOR_CLOCK_SIZE) {
    return false;
  }

  // Import's client ID must NOT be in op's clock (was pruned away)
  if (importClientId in opClock) {
    return false;
  }

  // Check shared keys between import and op
  const importKeys = Object.keys(importClock);
  const sharedKeys = importKeys.filter((k) => k in opClock);

  if (sharedKeys.length > 0) {
    // Shared keys exist: verify op has >= values for ALL (causal knowledge).
    // If the op has LESS knowledge for any shared key, this is genuine concurrency.
    for (const key of sharedKeys) {
      if (opClock[key] < importClock[key]) {
        return false;
      }
    }
    // All shared keys show op >= import → op inherited import's knowledge
    return true;
  }

  // No shared keys: clocks are completely disjoint.
  // This happens when the import has a single entry (e.g., {newClientId:1}) and
  // the op's MAX-size clock contains entirely different client IDs.
  //
  // We cannot determine causality from clocks alone, so we use UUIDv7 ordering
  // as additional evidence. UUIDv7 contains monotonic timestamps, so an op with
  // a later ID was created after the import (modulo negligible clock drift).
  return opId > importId;
};

@Injectable({
  providedIn: 'root',
})
export class SyncImportFilterService {
  private opLogStore = inject(OperationLogStoreService);

  /**
   * Filters out operations invalidated by a SYNC_IMPORT, BACKUP_IMPORT, or REPAIR.
   *
   * ## Clean Slate Semantics
   * Imports are explicit user actions to restore all clients to a specific state.
   * ALL operations without knowledge of the import are dropped - no exceptions.
   *
   * ## Vector Clock Comparison Results
   * | Comparison     | Meaning                              | Action  |
   * |----------------|--------------------------------------|---------|
   * | GREATER_THAN   | Op created after seeing import       | ✅ Keep |
   * | EQUAL          | Same causal history as import        | ✅ Keep |
   * | LESS_THAN      | Op dominated by import               | ❌ Filter|
   * | CONCURRENT     | Op created without knowledge of import| ❌ Filter|
   *
   * CONCURRENT ops are filtered even if they come from a client the import
   * didn't know about. This ensures a true "restore to point in time" semantic.
   *
   * The import can be in the current batch OR in the local store from a
   * previous sync cycle. We check both to handle the case where old ops from
   * another client arrive after we already downloaded the import.
   *
   * @param ops - Operations to filter (already migrated)
   * @returns Object with `validOps`, `invalidatedOps`, optionally `filteringImport`,
   *          and `isLocalUnsyncedImport` indicating if dialog should be shown
   */
  async filterOpsInvalidatedBySyncImport(ops: Operation[]): Promise<{
    validOps: Operation[];
    invalidatedOps: Operation[];
    filteringImport?: Operation;
    isLocalUnsyncedImport: boolean;
  }> {
    // Find full state import operations (SYNC_IMPORT, BACKUP_IMPORT, or REPAIR) in current batch
    const fullStateImportsInBatch = ops.filter(
      (op) =>
        op.opType === OpType.SyncImport ||
        op.opType === OpType.BackupImport ||
        op.opType === OpType.Repair,
    );

    // Check local store for previously downloaded import
    // Use getLatestFullStateOpEntry to get metadata (source, syncedAt)
    const storedEntry = await this.opLogStore.getLatestFullStateOpEntry();
    const storedImport = storedEntry?.op;

    // Determine the latest import (from batch or store)
    // Also track whether we're using the stored entry (needed for isLocalUnsyncedImport check)
    let latestImport: Operation | undefined;
    let usingStoredEntry = false;

    if (fullStateImportsInBatch.length > 0) {
      // Find the latest in the current batch
      const latestInBatch = fullStateImportsInBatch.reduce((latest, op) =>
        op.id > latest.id ? op : latest,
      );
      // Compare with stored import (if any)
      if (storedImport && storedImport.id > latestInBatch.id) {
        latestImport = storedImport;
        usingStoredEntry = true;
      } else {
        latestImport = latestInBatch;
        usingStoredEntry = false;
      }
    } else if (storedImport) {
      // No import in batch, but we have one from a previous sync
      latestImport = storedImport;
      usingStoredEntry = true;
    }

    // No imports found anywhere = no filtering needed
    if (!latestImport) {
      return { validOps: ops, invalidatedOps: [], isLocalUnsyncedImport: false };
    }

    // Determine if the filtering import is a local unsynced import.
    // This is used to decide whether to show the conflict dialog.
    //
    // isLocalUnsyncedImport is TRUE only when:
    // 1. We're using the stored entry (not a batch import)
    // 2. The stored entry was created locally (source='local')
    // 3. It hasn't been synced yet (no syncedAt)
    //
    // When true, the dialog SHOULD show - user must choose between their local
    // import and the remote data being filtered.
    //
    // When false (batch import, remote stored import, or synced local import),
    // the dialog should NOT show - old ops are silently discarded.
    const isLocalUnsyncedImport =
      usingStoredEntry &&
      !!storedEntry &&
      storedEntry.source === 'local' &&
      !storedEntry.syncedAt;

    OpLog.normal(
      `SyncImportFilterService: Filtering ops against SYNC_IMPORT from client ${latestImport.clientId} (op: ${latestImport.id})`,
    );
    OpLog.debug(
      `SyncImportFilterService: SYNC_IMPORT vectorClock: ${vectorClockToString(latestImport.vectorClock)}`,
    );

    // NORMALIZATION: If the local import clock exceeds MAX_VECTOR_CLOCK_SIZE, the server
    // stored a pruned version. Remote clients created ops based on the pruned version.
    // We must compare against the same pruned version to avoid false CONCURRENT results.
    // After pruning, some existing client IDs may be removed from the import clock.
    // This is intentional: isLikelyPruningArtifact will then correctly treat those
    // clients' ops as post-import ops with inherited knowledge (all shared keys >= import).
    const importClockForComparison =
      Object.keys(latestImport.vectorClock).length > MAX_VECTOR_CLOCK_SIZE
        ? limitVectorClockSize(latestImport.vectorClock, latestImport.clientId, [])
        : latestImport.vectorClock;

    const validOps: Operation[] = [];
    const invalidatedOps: Operation[] = [];

    for (const op of ops) {
      // Full state import operations themselves are always valid
      if (
        op.opType === OpType.SyncImport ||
        op.opType === OpType.BackupImport ||
        op.opType === OpType.Repair
      ) {
        validOps.push(op);
        continue;
      }

      // Use VECTOR CLOCK comparison instead of UUIDv7 timestamps.
      // Vector clocks track CAUSALITY ("did this client know about the import?")
      // rather than wall-clock time, making them immune to client clock drift.
      //
      // Clean Slate Semantics:
      // - GREATER_THAN: Op was created by a client that SAW the import → KEEP
      // - EQUAL: Same causal history as import → KEEP
      // - CONCURRENT: Op created WITHOUT knowledge of import → FILTER
      // - LESS_THAN: Op is dominated by import → FILTER
      //
      // CONCURRENT ops are filtered even from "unknown" clients. The import is
      // an explicit user action to restore to a specific state - any concurrent
      // work is intentionally discarded to ensure a clean slate.
      const comparison = compareVectorClocks(op.vectorClock, importClockForComparison);

      // DIAGNOSTIC LOGGING: Log vector clock comparison details
      // This helps debug issues where ops are incorrectly filtered as CONCURRENT
      OpLog.debug(
        `SyncImportFilterService: Comparing op ${op.id} (${op.actionType}) from client ${op.clientId}\n` +
          `  Op vectorClock:     ${vectorClockToString(op.vectorClock)}\n` +
          `  Import vectorClock: ${vectorClockToString(importClockForComparison)}` +
          (importClockForComparison !== latestImport.vectorClock
            ? ` (normalized from ${Object.keys(latestImport.vectorClock).length} entries)`
            : '') +
          `\n  Comparison result:  ${comparison}`,
      );

      if (
        comparison === VectorClockComparison.GREATER_THAN ||
        comparison === VectorClockComparison.EQUAL
      ) {
        // Op was created by a client that had knowledge of the import
        validOps.push(op);
      } else if (
        comparison === VectorClockComparison.CONCURRENT &&
        isLikelyPruningArtifact(op.vectorClock, op.clientId, importClockForComparison)
      ) {
        // Op appears CONCURRENT but is from a new client that inherited the import's
        // clock. Server-side pruning dropped an import entry, making the clocks look
        // concurrent when the op actually has full causal knowledge of the import.
        OpLog.normal(
          `SyncImportFilterService: KEEPING op ${op.id} (${op.actionType}) despite CONCURRENT comparison.\n` +
            `  Client ${op.clientId} not in import clock - new client after import.\n` +
            `  All shared vector clock keys are >= import values (pruning artifact).`,
        );
        validOps.push(op);
      } else if (
        comparison === VectorClockComparison.CONCURRENT &&
        op.clientId === latestImport.clientId &&
        (op.vectorClock[op.clientId] ?? 0) > (importClockForComparison[op.clientId] ?? 0)
      ) {
        // Op is from the SAME client that created the import, with a higher counter.
        // A client can't create ops concurrent with its own import — all post-import
        // ops from the import client necessarily have causal knowledge of the import.
        // Same-client counter comparison is definitive (monotonically increasing),
        // not heuristic. CONCURRENT here is always a pruning artifact from asymmetric
        // clock evolution (different entries pruned from op's evolving clock vs
        // import's frozen clock).
        OpLog.normal(
          `SyncImportFilterService: KEEPING op ${op.id} (${op.actionType}) despite CONCURRENT - same client as import.\n` +
            `  Client ${op.clientId} counter: op=${op.vectorClock[op.clientId]} > import=${importClockForComparison[op.clientId]} (post-import op).`,
        );
        validOps.push(op);
      } else if (
        comparison === VectorClockComparison.CONCURRENT &&
        isOpSidePruningArtifact(
          op.vectorClock,
          importClockForComparison,
          latestImport.clientId,
          op.id,
          latestImport.id,
        )
      ) {
        // Op's clock was pruned by the server, dropping the import's client ID.
        // This happens when the import has a small clock (e.g., from a fresh client
        // with {clientId:1}) and the op's established clock is at MAX_VECTOR_CLOCK_SIZE.
        // The server pruned the import client's low-counter entry from the op's clock,
        // making the op appear CONCURRENT when it was actually created after the import.
        OpLog.normal(
          `SyncImportFilterService: KEEPING op ${op.id} (${op.actionType}) despite CONCURRENT comparison.\n` +
            `  Op clock was pruned by server, losing import client ${latestImport.clientId}.\n` +
            `  Op clock size: ${Object.keys(op.vectorClock).length}, Import clock size: ${Object.keys(importClockForComparison).length}`,
        );
        validOps.push(op);
      } else {
        // CONCURRENT or LESS_THAN: Op was created without knowledge of import
        // Filter it to ensure clean slate semantics
        OpLog.warn(
          `SyncImportFilterService: FILTERING op ${op.id} (${op.actionType}) as ${comparison}\n` +
            `  Op vectorClock:     ${vectorClockToString(op.vectorClock)}\n` +
            `  Import vectorClock: ${vectorClockToString(importClockForComparison)}` +
            (importClockForComparison !== latestImport.vectorClock
              ? ` (normalized from ${Object.keys(latestImport.vectorClock).length} entries)`
              : '') +
            `\n  Import client:      ${latestImport.clientId}\n` +
            `  Op client:          ${op.clientId}`,
        );
        invalidatedOps.push(op);
      }
    }

    return {
      validOps,
      invalidatedOps,
      filteringImport: latestImport,
      isLocalUnsyncedImport,
    };
  }
}
