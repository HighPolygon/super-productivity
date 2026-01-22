# Data Subject Request Procedures (GDPR Articles 15-22)

**Organization:** Super Productivity Sync
**Data Controller:** Johannes Millan
**Contact:** contact@super-productivity.com
**Last Updated:** 2026-01-22
**Next Review:** 2027-01-22

---

## Purpose

This document defines standard operating procedures for handling data subject requests under GDPR Articles 15-22. It ensures:

- Timely response (within 1 month)
- Proper verification of requestor identity
- Complete and accurate responses
- Compliance with GDPR requirements

---

## Quick Reference

### Response Timeline

- **Acknowledgment:** Within 3 business days
- **Response:** Within 1 month (30 days) from receipt
- **Extension:** Up to 2 additional months if complex, with explanation to user

### Key Contacts

| Role                      | Email                          | Responsibility                         |
| ------------------------- | ------------------------------ | -------------------------------------- |
| **Data Subject Requests** | contact@super-productivity.com | Primary contact for all requests       |
| **Data Controller**       | Johannes Millan                | Final decision authority               |
| **Technical Support**     | [To be assigned]               | Data extraction, deletion verification |

---

## 1. Types of Data Subject Requests

### Article 15 - Right to Access

User requests copy of their personal data.

### Article 16 - Right to Rectification

User requests correction of inaccurate data.

### Article 17 - Right to Erasure ("Right to be Forgotten")

User requests deletion of their personal data.

### Article 18 - Right to Restriction of Processing

User requests limitation on how data is processed.

### Article 19 - Notification Obligation

Notify third parties of rectification, erasure, or restriction.

### Article 20 - Right to Data Portability

User requests data in machine-readable format for transfer to another service.

### Article 21 - Right to Object

User objects to processing based on legitimate interests.

### Article 22 - Rights Related to Automated Decision-Making

User objects to automated decisions (Not applicable to Super Sync).

---

## 2. Receiving and Logging Requests

### Request Channels

Requests can be received via:

- ✅ **Email:** contact@super-productivity.com (primary)
- ✅ **In-app:** Support ticket or settings page
- ⚠️ **Postal mail:** [Address if provided in privacy policy]
- ⚠️ **Phone:** [If support phone number provided]

### Initial Steps (Day 1)

**STEP 1: Log the Request**
Create entry in Data Subject Request Log (spreadsheet or database):

- Request ID (e.g., DSR-2026-001)
- Date received
- Request type (Access, Erasure, etc.)
- Requestor email
- Requestor name (if provided)
- Request details/description
- Status: "Received"

**STEP 2: Acknowledge Receipt**
Send acknowledgment email within 3 business days:

```
Subject: Data Subject Request Received - [Request ID]

Dear [Name / User],

We have received your data subject request regarding [your Super Productivity Sync account / data].

Request ID: [DSR-2026-XXX]
Request Type: [Access / Erasure / Rectification / etc.]
Received Date: [YYYY-MM-DD]

We will process your request within 30 days as required by GDPR. If we need additional information to verify your identity or process your request, we will contact you.

If you have questions, please reply to this email referencing Request ID [DSR-2026-XXX].

Best regards,
Super Productivity Sync Team
contact@super-productivity.com
```

**STEP 3: Assign Responsibility**

- Simple requests (e.g., account deletion): Can be handled by support team
- Complex requests (e.g., access to specific data categories): Escalate to data controller

---

## 3. Identity Verification

**Why Required:** Ensure we provide data only to the legitimate account holder (security).

### Standard Verification Methods

**Method 1: Email Verification (Preferred)**

- Request received from registered email address
- Send confirmation link to registered email
- User must click link to confirm identity

**Method 2: Account Login (For App-Based Requests)**

- User makes request while logged into account
- Authentication via JWT proves identity

**Method 3: Additional Information (If Needed)**

- Ask for account creation date
- Ask for last login date
- Ask for recent activity details (e.g., last task created)

### Verification Template

```
Subject: Identity Verification Required - [Request ID]

Dear [User],

To protect your privacy and security, we need to verify your identity before processing your data subject request.

Please confirm by replying to this email with:
1. Your registered email address
2. Approximate account creation date (month/year)
3. Any other details that can help us verify your identity

Alternatively, log into your Super Productivity account and submit the request via Settings > Privacy > Data Subject Request.

Thank you for your understanding.

Best regards,
Super Productivity Sync Team
```

### Red Flags (Potential Fraudulent Requests)

- Request from different email than registered
- Vague or suspicious request details
- Request for another person's data
- Unusual urgency or threats

**Action:** Request additional verification or decline with justification.

---

## 4. Processing Specific Request Types

## 4.1 Right to Access (Article 15)

### What User Gets

- Copy of all personal data we hold about them
- Information about processing (purposes, categories, recipients, retention)
- Source of data (user-provided)
- User's rights (rectification, erasure, etc.)

### Implementation

**STEP 1: Extract Data**
User can self-serve via app:

- Go to Settings > Sync > Export Data
- Generates JSON file with all sync operations, snapshots, and settings

Support team can also extract:

- Database query for user's operations
- Database query for user's account information
- Database query for user's devices

**STEP 2: Prepare Access Report**
Create document containing:

```markdown
# Data Subject Access Report

**Request ID:** DSR-2026-XXX
**User Email:** user@example.com
**Report Date:** YYYY-MM-DD

## 1. Personal Data We Hold

### Account Information

- Email address: [email]
- Account created: [date]
- Last login: [date]
- Account status: Active
- E2EE enabled: Yes/No
- Storage used: [X MB]

**Encryption Status:**

- Your account uses E2EE: ✅ Yes / ❌ No
- Data stored encrypted on server: [Yes if E2EE enabled, No otherwise]
- Note: Without E2EE enabled, your data is stored unencrypted in our PostgreSQL database

### Sync Data

- Number of operations: [count]
- Number of devices: [count]
- Latest snapshot date: [date]
  [Attached: sync-data-export.json]

**Note on Data Export:**

- E2EE users: Data exported in encrypted form (you must decrypt with your key)
- Non-E2EE users: Data exported in plaintext JSON format
- Exported data includes all sync operations and snapshots

### Server Logs (if still retained)

- Recent access times
- IP addresses (if within 7-14 day retention)
  [Note: Logs auto-deleted after 7-14 days]

## 2. Processing Information

**Purposes of Processing:**

- Synchronizing your productivity data across devices
- Account management and authentication
- Service security and abuse prevention

**Legal Basis:**

- Contract performance (Art. 6(1)(b))
- Legitimate interests for security (Art. 6(1)(f))

**Recipients:**

- Alfahosting GmbH (hosting provider, Germany, Data Processing Agreement in place)
- No third parties

**Retention Periods:**

- Sync operations: 45 days (covered by snapshots)
- Account data: Until account deletion
- Server logs: 7-14 days

**International Transfers:** None (all data in Germany)

**Your Rights:**

- Rectification: Update data via app
- Erasure: Delete account via app settings
- Restriction: Disable sync
- Portability: Export data (attached)
- Object: Contact us to object to processing
- Complain: Sächsischer Datenschutzbeauftragter (saechsdsb@slt.sachsen.de)

**Security Disclosure:**
Users should be informed that without E2EE enabled, their data is stored unencrypted in our PostgreSQL database. We recommend enabling E2EE for sensitive data.

## 3. Attachments

- sync-data-export.json (your productivity data)
```

**STEP 3: Send Response**
Email with:

- Access report (PDF or text)
- Data export (JSON file attachment)

**Timeline:** Within 1 month of request

---

## 4.2 Right to Erasure (Article 17)

### What Happens

- Account and all associated data deleted
- Includes: operations, snapshots, devices, account info
- Cascading delete in database

### Implementation

**STEP 1: Verify No Legal Obligation to Retain**
Check if any data must be retained:

- ❌ Tax records: NOT APPLICABLE (no paid accounts yet)
- ❌ Legal claims: NOT APPLICABLE (no ongoing disputes)
- ✅ OK to delete

**STEP 2: User Can Self-Serve**
In app:

- Settings > Account > Delete Account
- Confirmation dialog
- Immediate deletion

OR Support team uses CLI:

```bash
npm run delete-user -- user@example.com
```

**STEP 3: Verify Deletion**
Check database:

```sql
SELECT * FROM users WHERE email = 'user@example.com';
-- Should return no results
```

**STEP 4: Send Confirmation**

```
Subject: Account Deletion Confirmation - [Request ID]

Dear [User],

Your Super Productivity Sync account has been deleted as requested.

Deleted:
- Account information (email, password hash)
- All sync operations and snapshots
- All device records
- All associated data

Timeline:
- Database deletion: Immediate
- Backup purge: Within 7 days (standard backup rotation)

Your data is no longer accessible and will be fully purged from all systems within 7 days.

If you created a new account with the same email, it will be treated as a new account with no connection to the deleted account.

Thank you for using Super Productivity Sync.

Best regards,
Super Productivity Sync Team
```

**Timeline:** Immediate (acknowledge within 1 month)

### Exceptions to Erasure Right

Decline erasure request if:

- Legal obligation to retain data (e.g., tax records for paid accounts)
- Necessary for legal claims or defense
- Public interest (Not applicable to Super Sync)

**Template for Declining:**

```
We cannot delete your data at this time because [reason, e.g., "we are legally required to retain billing records for 10 years under German tax law"].

However, we can restrict processing of your data. Please let us know if you would like to proceed with restriction instead of erasure.
```

---

## 4.3 Right to Rectification (Article 16)

### What User Can Do

Correct inaccurate or incomplete data.

### Implementation

**STEP 1: Identify Data to Correct**

- Email address
- User content (tasks, projects, notes)

**STEP 2: User Can Self-Serve**
For most data:

- User updates data directly in app (sync operations handle corrections)

For email address change:

- [To be implemented: Email change feature with verification]
- Currently: Delete account and re-register (workaround)

**STEP 3: Send Confirmation**

```
Subject: Data Rectification Completed - [Request ID]

Dear [User],

We have corrected your personal data as requested.

Updated:
- [List what was corrected]

Your updated data is now reflected in your account.

Best regards,
Super Productivity Sync Team
```

**Timeline:** Within 1 month

---

## 4.4 Right to Restriction of Processing (Article 18)

### What Happens

Limit processing while dispute resolved or user objects.

### Implementation

**STEP 1: Assess Reason for Restriction**

- User contests accuracy (restrict while verifying)
- Processing unlawful but user doesn't want erasure
- Controller no longer needs data but user needs it for legal claim
- User objects to processing (restrict while assessing objection)

**STEP 2: Implement Restriction**

**Option 1: User Disables Sync**

- User turns off sync in app settings
- Data remains stored but not actively processed

**Option 2: Support Team Flags Account**

- Add "restricted" flag to user account in database
- Document reason for restriction
- No new sync operations processed

**STEP 3: Send Confirmation**

```
Subject: Processing Restriction Applied - [Request ID]

Dear [User],

We have restricted processing of your personal data as requested.

Restriction Details:
- Sync operations: Paused
- Data storage: Retained but not actively processed
- Duration: Until [condition, e.g., "dispute resolved"]

You can reactivate processing by [action, e.g., "re-enabling sync in settings"].

Best regards,
Super Productivity Sync Team
```

**Timeline:** Within 1 month

---

## 4.5 Right to Data Portability (Article 20)

### What User Gets

Machine-readable copy of data for transfer to another service.

### Implementation

**STEP 1: Export Data**
Same as Access request:

- User exports via Settings > Sync > Export Data
- JSON format (machine-readable)
- Includes: operations, snapshots, settings

**STEP 2: Send Data**
Email with JSON file attachment.

**Note:** Data format is Super Sync's internal format. User may need to convert for use with other services.

**Timeline:** Within 1 month

---

## 4.6 Right to Object (Article 21)

### What User Can Do

Object to processing based on legitimate interests.

### Implementation

**STEP 1: Assess Objection**

- Is processing based on legitimate interests (Art. 6(1)(f))? → Yes for server logs
- Does objection relate to direct marketing? → NOT APPLICABLE (no marketing)

**STEP 2: Evaluate Objection**

- Can we demonstrate compelling legitimate grounds that override user's interests?
- Example: Security logs needed to prevent fraud

**STEP 3: Respond**

If objection accepted:

```
We have stopped processing your data for [specific purpose].

Actions taken:
- [List actions, e.g., "Stopped logging your IP address"]
```

If objection declined:

```
We must continue processing your data for [specific purpose] because [compelling legitimate grounds, e.g., "security logs are necessary to prevent fraud and unauthorized access, which protects all users including yourself"].

You have the right to lodge a complaint with the Sächsischer Datenschutzbeauftragter (saechsdsb@slt.sachsen.de) if you disagree with this decision.
```

**Timeline:** Within 1 month

---

## 4.7 Notification Obligation (Article 19)

### What It Means

If we rectify, erase, or restrict data, we must notify recipients of that data.

### Implementation

**STEP 1: Identify Recipients**
For Super Sync:

- Alfahosting (hosting provider) - changes automatically reflected
- No other recipients

**STEP 2: Notify Recipients** (If Applicable)
Not required for Alfahosting (automated cascading delete).

If third-party integrations added in future:

- Email third parties: "User [X] has requested data deletion, please remove from your systems"
- Track confirmation of deletion

**Currently:** NO ACTION NEEDED (no third-party recipients)

---

## 5. Response Templates

### Template 1: Acknowledgment

```
Subject: Data Subject Request Received - [Request ID]

Dear [User],

We have received your [type of request] request.

Request ID: [DSR-2026-XXX]
Received: [Date]
Expected Response: Within 30 days ([Date])

We will contact you if we need additional information to process your request.

Best regards,
Super Productivity Sync Team
```

### Template 2: Request for Additional Information

```
Subject: Additional Information Needed - [Request ID]

Dear [User],

To process your request, we need the following information:
- [List required information]

Please reply to this email with the requested details.

Note: The 30-day response period is paused until we receive this information.

Best regards,
Super Productivity Sync Team
```

### Template 3: Extension Notification (Complex Requests)

```
Subject: Processing Time Extension - [Request ID]

Dear [User],

Your request is complex and requires additional time to process.

We are extending the response period by [1-2 months] as permitted under GDPR Article 12(3).

Reason for extension: [e.g., "requires extensive data retrieval and anonymization"]

New expected response date: [Date]

We apologize for the delay and appreciate your patience.

Best regards,
Super Productivity Sync Team
```

### Template 4: Request Declined

```
Subject: Data Subject Request - Unable to Process - [Request ID]

Dear [User],

We are unable to process your request for the following reason:

[Reason, e.g., "We cannot verify your identity" or "We are legally required to retain this data"]

[Explanation of reason]

Your rights:
You have the right to lodge a complaint with the supervisory authority (Sächsischer Datenschutzbeauftragter) if you believe our decision is incorrect.

Contact: saechsdsb@slt.sachsen.de
Website: https://www.saechsdsb.de/

Best regards,
Super Productivity Sync Team
```

---

## 6. Request Log

Maintain a spreadsheet or database table with:

| Field               | Description           | Example                                    |
| ------------------- | --------------------- | ------------------------------------------ |
| Request ID          | Unique identifier     | DSR-2026-001                               |
| Date Received       | When request arrived  | 2026-01-22                                 |
| Request Type        | Type of request       | Access, Erasure, etc.                      |
| Requestor Email     | User's email          | user@example.com                           |
| Status              | Current status        | Received, In Progress, Completed, Declined |
| Assigned To         | Person handling       | Johannes Millan                            |
| Verification Method | How identity verified | Email confirmation                         |
| Response Date       | When response sent    | 2026-02-15                                 |
| Days to Response    | Time taken            | 24 days                                    |
| Outcome             | Result                | Completed, Declined, etc.                  |
| Notes               | Additional details    | -                                          |

**Retention:** Keep log for 3 years (accountability).

---

## 7. Service Level Agreement (SLA)

### Internal Performance Targets

| Milestone            | Target           | GDPR Requirement          |
| -------------------- | ---------------- | ------------------------- |
| **Acknowledgment**   | 3 business days  | No specific requirement   |
| **Response**         | 14 days (target) | 1 month (max)             |
| **Complex requests** | 30 days          | 3 months (with extension) |
| **Verification**     | 2 business days  | Reasonable time           |

### Monitoring

**Monthly Review:**

- Count of requests received
- Average response time
- Percentage meeting 14-day target
- Percentage meeting 30-day requirement
- Any declined requests and reasons

**Quarterly Report to Data Controller:**

- Summary of all requests
- Trends or patterns
- Process improvements needed

---

## 8. Training and Resources

### Staff Training

All personnel handling data subject requests must understand:

- GDPR data subject rights (Articles 15-22)
- Identity verification procedures
- Response timelines and SLA
- How to extract user data
- How to use response templates
- Escalation procedures

### Training Frequency

- Initial training: Before handling requests
- Annual refresher: Every year
- Update training: When procedures change

### Quick Reference Guide

Provide one-page summary:

- Request types and response times
- Verification steps
- Common templates
- Escalation contacts

---

## 9. Escalation Procedures

### When to Escalate

- Complex legal questions (e.g., conflicting legal obligations)
- Requests involving minors or vulnerable individuals
- Requests from third parties (e.g., law enforcement)
- Requests that seem fraudulent or abusive
- User threatens legal action or complaint
- Unclear how to respond

### Escalation Contact

**Data Controller:** Johannes Millan (contact@super-productivity.com)

**Legal Counsel:** [To be assigned if needed]

---

## 10. Continuous Improvement

### Process Review

**Annually review:**

- Request volume and trends
- Response times (meeting SLA?)
- Common issues or complaints
- Template effectiveness
- Staffing adequacy

**Update procedures if:**

- GDPR guidance changes
- Supervisory authority issues new guidance
- Frequent issues identified
- Technology changes (e.g., new data categories)

---

## 11. Appendices

### Appendix A: GDPR Articles Quick Reference

**Article 12:** Transparent communication, 1-month response time
**Article 15:** Right to access
**Article 16:** Right to rectification
**Article 17:** Right to erasure
**Article 18:** Right to restriction
**Article 19:** Notification obligation
**Article 20:** Right to data portability
**Article 21:** Right to object
**Article 22:** Automated decision-making (Not applicable)

### Appendix B: Supervisory Authority Contact

**Sächsischer Datenschutzbeauftragter** (Saxony Data Protection Authority)

- Website: https://www.saechsdsb.de/
- Complaint form: https://www.saechsdsb.de/beschwerde
- Email: saechsdsb@slt.sachsen.de
- Phone: +49 351 85471-101

### Appendix C: CLI Commands for Data Operations

**Export user data:**

```bash
npm run export-user -- user@example.com
```

**Delete user account:**

```bash
npm run delete-user -- user@example.com
```

**Verify user deletion:**

```bash
npm run verify-deletion -- user@example.com
```

_(Actual command names to be confirmed)_

---

## Approval

**Data Controller:** **\*\***\*\***\*\***\_**\*\***\*\***\*\*** Date: \***\*\_\_\_\*\***
_Johannes Millan_

**Next Review:** 2027-01-22

---

_This document is for internal use. Maintain confidentiality._
