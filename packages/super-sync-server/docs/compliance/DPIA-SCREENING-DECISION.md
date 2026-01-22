# Data Protection Impact Assessment (DPIA) Screening Decision

**Service:** Super Productivity Sync
**Data Controller:** Johannes Millan
**Assessment Date:** 2026-01-22
**Assessed By:** Data Controller
**Next Review:** 2027-01-22 or when processing significantly changes

---

## Executive Summary

**Decision:** Full DPIA **NOT REQUIRED**

**Justification:** Super Productivity Sync's data processing does not meet the GDPR Article 35 thresholds for mandatory DPIA. Screening assessment shows low risk to data subjects' rights and freedoms.

**Confidence:** High (based on WP248 Rev.01 criteria and GDPR Article 35)

---

## 1. Legal Background

### GDPR Article 35(1) - When is DPIA Required?

A DPIA is required when processing is:

> "likely to result in a high risk to the rights and freedoms of natural persons"

### GDPR Article 35(3) - Mandatory DPIA Scenarios

DPIA is **required** for:

1. **Systematic and extensive evaluation** based on automated processing (including profiling) with legal or similarly significant effects
2. **Large-scale processing** of special categories of data (Art. 9) or criminal convictions (Art. 10)
3. **Systematic monitoring** of publicly accessible areas on a large scale

### WP248 Rev.01 - DPIA Criteria

The Article 29 Working Party (now EDPB) identified 9 criteria. If processing meets **2 or more**, DPIA is likely required:

1. Evaluation or scoring
2. Automated decision-making with legal/significant effects
3. Systematic monitoring
4. Sensitive data or data of highly personal nature
5. Data processed on a large scale
6. Matching or combining datasets
7. Data concerning vulnerable data subjects
8. Innovative use or new technological solutions
9. Processing that prevents data subjects from exercising a right or using a service

---

## 2. Screening Assessment

### Question 1: Does processing involve systematic and extensive evaluation/profiling with legal or significant effects?

**Answer:** ❌ NO

**Analysis:**

- Super Sync stores and synchronizes user productivity data (tasks, projects, notes)
- No automated decision-making performed on this data
- No profiling of user behavior or characteristics
- No scoring, evaluation, or prediction algorithms
- Users manually create and manage all data

**Conclusion:** This criterion is NOT met.

---

### Question 2: Does processing involve large-scale special category data?

**Answer:** ❌ NO

**Analysis:**

**Special Categories (Art. 9):**

- ❌ Racial or ethnic origin
- ❌ Political opinions
- ❌ Religious or philosophical beliefs
- ❌ Trade union membership
- ❌ Genetic data
- ❌ Biometric data (for unique identification)
- ❌ Health data
- ❌ Sex life or sexual orientation

**Super Sync Processes:**

- ✅ Email addresses (regular personal data)
- ✅ Password hashes (security data, not special category)
- ✅ User-created tasks, notes, time tracking (regular personal data)

**User Responsibility:**
Users _may_ choose to store sensitive information (e.g., "doctor appointment" note), but:

- This is user's choice, not required by service
- We recommend E2EE for sensitive data
- Not "large-scale processing" of special categories
- Incidental, not systematic

**Scale:**

- Small to medium user base (not "large-scale" per EDPB guidelines)
- No systematic collection of special category data

**Conclusion:** This criterion is NOT met.

---

### Question 3: Does processing involve systematic monitoring of publicly accessible areas on a large scale?

**Answer:** ❌ NO

**Analysis:**

- Super Sync is a private sync service for user's own data
- No monitoring of public areas (physical or virtual)
- No surveillance, tracking, or observation of individuals
- Users only access their own data
- No third-party tracking or analytics

**Conclusion:** This criterion is NOT met.

---

### Additional Consideration: Encryption at Rest

**Current State:** ⚠️ Database encryption at rest NOT implemented

**Analysis:**

- PostgreSQL data files stored unencrypted on disk
- Compensating control: Optional E2EE available (user choice)
- Risk: Users who don't enable E2EE have unencrypted data at rest

**Impact on DPIA Decision:**

- Does NOT change DPIA requirement (still not required per Art. 35)
- Encryption gap increases risk severity but doesn't trigger DPIA thresholds
- Risk is addressed through:
  - Optional E2EE offering
  - Physical security at hosting provider (Alfahosting)
  - Access controls (JWT authentication, rate limiting)
  - Transparent disclosure in privacy policy

**Conclusion:**
The lack of database encryption at rest is documented as a compliance gap (85% vs 92%),
but does not meet the "high risk" threshold requiring a full DPIA under Art. 35.

---

## 3. WP248 Criteria Assessment

Evaluating against all 9 criteria from WP248 Rev.01:

| #   | Criterion                                                    | Met?       | Justification                                                                                  |
| --- | ------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------- |
| 1   | **Evaluation or scoring**                                    | ❌ No      | No automated evaluation of user characteristics or behavior                                    |
| 2   | **Automated decision-making with legal/significant effects** | ❌ No      | No automated decisions made; user manually manages all data                                    |
| 3   | **Systematic monitoring**                                    | ❌ No      | No monitoring of user behavior; no tracking or surveillance                                    |
| 4   | **Sensitive data or highly personal nature**                 | ⚠️ Partial | User content may be personal (tasks, notes), but not systematically sensitive. E2EE available. |
| 5   | **Large-scale processing**                                   | ❌ No      | Small to medium user base; not multinational or millions of users                              |
| 6   | **Matching or combining datasets**                           | ❌ No      | User's data stays separate; no cross-referencing with external datasets                        |
| 7   | **Vulnerable data subjects**                                 | ❌ No      | General user base, not targeting children or vulnerable populations                            |
| 8   | **Innovative use or new technology**                         | ❌ No      | Standard sync technology; operation log-based sync is established practice                     |
| 9   | **Processing prevents exercising rights**                    | ❌ No      | Opt-in service; users can export, delete, disable sync at any time                             |

**Result:** Only 0.5 criteria met (partial on #4)

**Threshold:** 2 or more criteria = DPIA likely required

**Conclusion:** Below threshold, DPIA NOT required.

---

## 4. Risk Assessment

### Risk Level: LOW-MEDIUM

**Overall Risk Assessment:**

- E2EE users: LOW risk (zero-knowledge encryption)
- Non-E2EE users: MEDIUM risk (unencrypted data at rest)
- Mitigations: Physical security, access controls, optional E2EE

| Risk Factor           | Assessment | Justification                                                       |
| --------------------- | ---------- | ------------------------------------------------------------------- |
| **Data Sensitivity**  | Low-Medium | Regular personal data (tasks, notes); user controls sensitivity     |
| **Data Volume**       | Low-Medium | Per-user data volume modest; no aggregation across users            |
| **Identifiability**   | Low        | Email only identifier; no linkage to external databases             |
| **Automation**        | None       | No automated profiling, scoring, or decision-making                 |
| **Transparency**      | High       | Open source, clear privacy policy, user controls                    |
| **Consent & Control** | High       | Opt-in service, user can delete/export/disable anytime              |
| **Security**          | High       | E2EE optional, HTTPS, bcrypt, rate limiting, no third-party sharing |
| **Reversibility**     | High       | User can export data, delete account, disable sync                  |

### User Harm Scenarios (Hypothetical)

| Scenario                     | Likelihood | Impact | Risk Level                             |
| ---------------------------- | ---------- | ------ | -------------------------------------- |
| Data breach (unencrypted)    | Low        | Medium | Low-Medium (MEDIUM for non-E2EE users) |
| Data breach (E2EE enabled)   | Low        | Low    | Very Low                               |
| Unauthorized account access  | Low        | Medium | Low-Medium                             |
| Service unavailability       | Medium     | Low    | Low                                    |
| Accidental data loss         | Low        | Medium | Low-Medium                             |
| Privacy violation (tracking) | None       | N/A    | None (no tracking implemented)         |

**Overall Risk to Rights and Freedoms:** LOW-MEDIUM (varies by E2EE usage)

---

## 5. Mitigating Factors

Super Productivity Sync implements multiple safeguards that reduce risk:

1. **Privacy by Design:**
   - Optional E2EE with zero-knowledge architecture
   - No analytics, tracking, or behavioral monitoring
   - Minimal data collection (data minimization)
   - Local-first architecture (sync optional)

2. **User Control:**
   - Users opt-in to sync service
   - Can disable sync at any time
   - Can delete account with full data purge
   - Can export all data (portability)
   - Can choose E2EE level (plaintext or encrypted)

3. **Technical Security:**
   - HTTPS/TLS encryption in transit
   - bcrypt password hashing (12 rounds)
   - JWT authentication with token revocation
   - Rate limiting and account lockout protection
   - Input validation and sanitization
   - SQL injection prevention (Prisma ORM)
   - Security headers (CSP, HSTS, etc.)

4. **Organizational Measures:**
   - Clear privacy policy (GDPR-compliant)
   - Data Processing Agreement with hosting provider
   - Incident response plan in place
   - Regular security updates
   - Compliance documentation maintained

5. **Transparency:**
   - Open source codebase (auditable)
   - No hidden data collection
   - Clear terms of service
   - Data subject rights honored

---

## 6. Comparative Analysis

### Similar Services (Benchmarking)

| Service Type                                       | DPIA Required? | Reason                                                 |
| -------------------------------------------------- | -------------- | ------------------------------------------------------ |
| **Sync Services (Dropbox, Google Drive)**          | Generally No   | Unless large-scale monitoring or special category data |
| **Productivity Apps (Todoist, Notion)**            | Generally No   | Similar use case to Super Sync                         |
| **Social Media (Facebook, Instagram)**             | **Yes**        | Large-scale profiling, targeted advertising            |
| **Health Apps (MyFitnessPal, health trackers)**    | **Yes**        | Special category data (health)                         |
| **Financial Services (Banks, credit scoring)**     | **Yes**        | Automated decisions with legal effects                 |
| **Employee Monitoring (keyloggers, surveillance)** | **Yes**        | Systematic monitoring, power imbalance                 |
| **Smart Home (cameras, IoT)**                      | Often Yes      | Systematic monitoring of private spaces                |

**Super Sync aligns with:** Productivity and sync services (DPIA not typically required)

---

## 7. Decision

### DPIA Status: **NOT REQUIRED**

**Justification:**

1. Does not meet GDPR Article 35(3) mandatory DPIA scenarios
2. Meets fewer than 2 WP248 criteria (threshold for likely DPIA requirement)
3. Processing poses low-medium risk to data subjects' rights and freedoms
4. Extensive mitigating factors in place (privacy by design, security, user control)
5. Comparable to other productivity/sync services that do not require DPIA

**Encryption Gap Consideration:**
While database encryption at rest is not currently implemented, this does not elevate
the processing to "high risk" requiring a DPIA. The service offers optional E2EE
as a compensating control, maintains transparent disclosure, and processes data on
a relatively small scale with strong access controls. Users requiring maximum security
are advised to enable E2EE.

**Documented Screening:** This document serves as evidence of DPIA screening for accountability purposes (Art. 5(2) GDPR).

---

## 8. Re-evaluation Triggers

A full DPIA **must** be conducted if processing changes significantly, such as:

### High Priority Triggers (Require DPIA)

- [ ] **Large-scale behavioral profiling** (e.g., analyzing user productivity patterns)
- [ ] **Automated decision-making** with legal/significant effects (e.g., credit scoring based on task data)
- [ ] **Systematic monitoring** of user behavior (e.g., detailed analytics, tracking)
- [ ] **Processing special category data** at scale (e.g., health tracking feature)
- [ ] **Biometric authentication** beyond passkeys (e.g., facial recognition, fingerprint analysis)
- [ ] **AI/ML features** analyzing user content (e.g., sentiment analysis, task prioritization)
- [ ] **Combining datasets** from external sources (e.g., linking with social media, employer data)
- [ ] **Targeting vulnerable populations** (e.g., version for children, medical patients)

### Medium Priority Triggers (Re-assess Need for DPIA)

- [ ] **User base grows significantly** (>100,000 active users)
- [ ] **New data categories** collected (e.g., location data, voice recordings)
- [ ] **Third-party integrations** sharing user data (e.g., API access for third-party apps)
- [ ] **International data transfers** outside EU (e.g., US-based hosting)
- [ ] **New recipients** of data (e.g., analytics providers, advertisers)
- [ ] **Monetization changes** (e.g., advertising, data selling)

### Low Priority Triggers (Monitor, Likely No DPIA Needed)

- [ ] Minor UI/UX changes
- [ ] Performance optimizations
- [ ] Bug fixes
- [ ] Additional sync providers (similar to existing)
- [ ] Implementation of mandatory encryption (would reduce risk, strengthen compliance)
- [ ] Removal of E2EE option (would increase risk, may trigger DPIA need)

**Process:**

1. Identify processing change
2. Re-run this screening assessment
3. If 2+ WP248 criteria met or Article 35(3) triggered → Conduct full DPIA
4. Document decision and update this file

---

## 9. Documentation and Accountability

### Records Maintained

This screening decision is part of GDPR accountability (Art. 5(2)):

- **This document:** DPIA screening decision
- **Records of Processing Activities:** Detailed processing documentation (Art. 30)
- **Privacy policy:** User-facing transparency
- **Code repository:** Technical implementation (open source)
- **Incident response plan:** Breach preparedness

### Internal Review Process

**Annual Review:**

- Review this screening decision annually
- Check for processing changes that trigger DPIA requirement
- Update decision if circumstances change

**Change-Triggered Review:**

- Any significant feature addition must be assessed
- Product manager/developer must flag potential DPIA triggers
- Data controller makes final DPIA necessity decision

**Next Review Date:** 2027-01-22

---

## 10. Approval and Sign-Off

**Screened By:**

- Name: Johannes Millan
- Role: Data Controller
- Date: 2026-01-22
- Signature: **\*\***\*\***\*\***\_**\*\***\*\***\*\***

**Decision:**
☑ DPIA NOT REQUIRED (screening documented)
☐ DPIA REQUIRED (proceed to full assessment)

**Justification Summary:**
Super Productivity Sync processes regular personal data (tasks, notes, email addresses) for synchronization purposes. The service does not involve automated decision-making, profiling, large-scale special category data processing, or systematic monitoring. Risk to data subjects is low due to optional E2EE, strong security measures, user control, and data minimization. Processing does not meet GDPR Article 35(3) thresholds or 2+ WP248 criteria. Therefore, a full DPIA is not required at this time.

**Re-evaluation Required If:**
Significant processing changes (see Section 8 triggers)

---

## 11. Appendices

### Appendix A: GDPR Article 35 Full Text (Excerpt)

> **Article 35 - Data protection impact assessment**
>
> 1. Where a type of processing in particular using new technologies, and taking into account the nature, scope, context and purposes of the processing, is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out an assessment of the impact of the envisaged processing operations on the protection of personal data. A single assessment may address a set of similar processing operations that present similar high risks.
> 2. A data protection impact assessment referred to in paragraph 1 shall in particular be required in the case of:
>    (a) a systematic and extensive evaluation of personal aspects relating to natural persons which is based on automated processing, including profiling, and on which decisions are based that produce legal effects concerning the natural person or similarly significantly affect the natural person;
>    (b) processing on a large scale of special categories of data referred to in Article 9(1), or of personal data relating to criminal convictions and offences referred to in Article 10; or
>    (c) a systematic monitoring of a publicly accessible area on a large scale.

### Appendix B: WP248 Rev.01 Criteria (Summary)

Source: Article 29 Working Party, Guidelines on Data Protection Impact Assessment (WP248 Rev.01)

**9 Criteria for DPIA Consideration:**

1. Evaluation or scoring
2. Automated decision-making with legal/significant effects
3. Systematic monitoring
4. Sensitive data or data of highly personal nature
5. Data processed on a large scale
6. Matching or combining datasets
7. Data concerning vulnerable data subjects
8. Innovative use or new technological solutions
9. Processing that prevents data subjects from exercising a right

**Threshold:** 2 or more criteria → DPIA likely required

### Appendix C: Change Log

| Date       | Change                     | Reason                        | Reviewed By     |
| ---------- | -------------------------- | ----------------------------- | --------------- |
| 2026-01-22 | Initial screening decision | GDPR compliance documentation | Johannes Millan |
|            |                            |                               |                 |

---

_This document is confidential and for internal use. May be requested by supervisory authority during audits._
