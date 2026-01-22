# Data Breach Incident Response Playbook

**Organization:** Super Productivity Sync
**Data Controller:** Johannes Millan
**Last Updated:** 2026-01-22
**Review Frequency:** Annual

---

## Purpose

This playbook provides step-by-step procedures for responding to data breaches in compliance with GDPR Articles 33 and 34. It ensures:

- Timely detection and response to security incidents
- Compliance with 72-hour breach notification requirement (GDPR Art. 33)
- Proper notification to affected users when required (GDPR Art. 34)
- Documentation for accountability

---

## Quick Reference

### GDPR Timeline Requirements

- **72 hours**: Maximum time to notify supervisory authority (from discovery)
- **Without undue delay**: Notify affected users if high risk to their rights
- **Immediately**: Begin containment and investigation

### Key Contacts

| Role            | Contact          | Phone          | Email                          |
| --------------- | ---------------- | -------------- | ------------------------------ |
| Data Controller | Johannes Millan  | [To be filled] | contact@super-productivity.com |
| Technical Lead  | [To be assigned] | [To be filled] | [To be filled]                 |
| Legal Counsel   | [To be assigned] | [To be filled] | [To be filled]                 |

### Supervisory Authority

**Sächsischer Datenschutzbeauftragter** (Saxony Data Protection Authority)

- Website: https://www.saechsdsb.de/
- Online reporting: https://www.saechsdsb.de/beschwerde
- Email: saechsdsb@slt.sachsen.de
- Phone: +49 351 85471-101
- Address: Devrientstraße 5, 01067 Dresden, Germany

---

## 1. What Constitutes a Data Breach?

A personal data breach is:

> "A breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorized disclosure of, or access to, personal data transmitted, stored or otherwise processed."

### Examples of Breaches:

- ✅ Unauthorized access to user accounts or database
- ✅ Data exposed publicly (misconfigured server, public bucket)
- ✅ Stolen backup containing user data
- ✅ Ransomware encrypting user data
- ✅ Accidental deletion of user data
- ✅ Data sent to wrong recipient
- ✅ Lost/stolen device or server disk containing unencrypted user data
- ✅ Physical server compromise or disk theft (data stored unencrypted for non-E2EE users)
- ✅ Successful SQL injection or XSS attack exposing user data
- ✅ Compromised credentials allowing unauthorized database access

### NOT a Breach (Usually):

- ❌ User forgot their password (no data exposed)
- ❌ DDoS attack (availability only, no data access)
- ❌ Failed login attempts (no data accessed)
- ❌ Patch or update causing temporary service disruption
- ❌ User error (user deletes their own data)

**When in doubt, treat it as a breach and investigate.**

---

## 2. Incident Detection

### Automated Monitoring

Currently, Super Sync relies on:

- Server logs (7-14 day retention)
- Failed login tracking (rate limiting)
- Health check endpoint (`/health`)

### Manual Detection Signs

Watch for:

- Unusual database queries in logs
- Spike in failed authentication attempts
- Reports from users about unauthorized access
- Unexpected data exports or large data transfers
- Server performance anomalies
- Security researcher reports
- Third-party notifications (Alfahosting, GitHub)

### Reporting a Suspected Breach

Anyone who suspects a breach should immediately:

1. Email: contact@super-productivity.com with subject "SECURITY INCIDENT"
2. Include: Date/time, what was observed, affected systems
3. Do NOT discuss publicly until assessed

---

## 3. Initial Response (Hour 0)

### Immediate Actions (Within 1 Hour of Discovery)

**STEP 1: Acknowledge and Log**

- [ ] Note discovery time (this starts the 72-hour clock)
- [ ] Create incident log file: `INCIDENT-[DATE]-[ID].md`
- [ ] Log format:
  ```
  Discovery Time: [YYYY-MM-DD HH:MM UTC]
  Reported By: [Name]
  Initial Description: [Brief summary]
  Systems Affected: [List]
  ```

**STEP 2: Assemble Response Team**

- [ ] Notify Data Controller (Johannes Millan)
- [ ] Notify Technical Lead
- [ ] Notify Legal Counsel (if applicable)
- [ ] Assign Incident Commander

**STEP 3: Initial Containment**

- [ ] If credentials compromised: Rotate immediately
- [ ] If database exposed: Restrict access to authorized IPs only
- [ ] If server compromised: Consider taking offline (balance with availability)
- [ ] Preserve evidence (logs, database snapshots, system state)

**STEP 4: Preliminary Assessment**

- [ ] Is this a confirmed breach or suspected breach?
- [ ] What data is potentially affected?
- [ ] How many users potentially affected?
- [ ] Is data encrypted?
  - ✅ E2EE enabled: Data encrypted client-side (low risk)
  - ❌ E2EE disabled: Data stored unencrypted in PostgreSQL (high risk)
  - ⚠️ IMPORTANT: E2EE is optional, not mandatory - verify per user
- [ ] Is breach ongoing or contained?

---

## 4. Investigation and Risk Assessment (Hours 1-24)

### Evidence Gathering

- [ ] Collect server logs (access logs, error logs, auth logs)
- [ ] Database query logs (if available)
- [ ] Review recent database changes
- [ ] Check for unauthorized user accounts
- [ ] Review recent code deployments
- [ ] Interview personnel with access

### Determine Breach Scope

Answer these questions:

**What data was affected?**

- [ ] Email addresses
- [ ] Password hashes
- [ ] Sync operations (encrypted if E2EE enabled, otherwise plaintext)
- [ ] Snapshots (encrypted if E2EE enabled, otherwise plaintext)
- [ ] Database files (always unencrypted at disk level)
- [ ] User content (tasks, projects, notes)
- [ ] IP addresses / logs
- [ ] Authentication tokens

**How many users affected?**

- [ ] All users
- [ ] Specific user subset (which?)
- [ ] Unknown (assume worst case)

**How was data accessed?**

- [ ] Unauthorized database access
- [ ] Application vulnerability (SQL injection, XSS, etc.)
- [ ] Stolen credentials
- [ ] Misconfiguration
- [ ] Physical theft
- [ ] Social engineering
- [ ] Other: **\*\*\*\***\_**\*\*\*\***

**When did breach occur?**

- [ ] Exact time/date if known
- [ ] Estimated timeframe
- [ ] Unclear (ongoing investigation)

### Risk Assessment for Users

**High Risk to Users' Rights** (MUST notify users):

- Unencrypted sensitive data exposed (tasks/notes of non-E2EE users containing personal info)
- Database backup stolen/exposed (non-E2EE user data readable in plaintext)
- Credentials compromised (password hashes stolen)
- Financial data exposed (payment info)
- Identity theft risk
- Discrimination risk

**Low Risk to Users' Rights** (May not require user notification):

- Data already encrypted with E2EE (server compromise doesn't expose plaintext)
- Note: Only applies to users who enabled E2EE (optional, not default)
- Only metadata exposed (IP addresses, timestamps)
- Already publicly available information
- Highly unlikely to cause harm

**Assessment Criteria:**
| Factor | High Risk | Low Risk |
|--------|-----------|----------|
| Data Type | Personal/financial | Technical metadata |
| Encryption | E2EE Disabled (Unencrypted in PostgreSQL) | E2EE Enabled (Encrypted client-side) |
| User Count | >100 users | <10 users |
| Misuse Likelihood | High | Low |
| User Harm Severity | Significant | Minimal |

Document risk assessment clearly - supervisory authority will review.

---

## 5. Notification to Supervisory Authority (Within 72 Hours)

### Decision Point: Must We Notify?

**YES - Notify if:**

- Confirmed breach of personal data occurred
- Risk to users' rights and freedoms (even if low)
- Any doubt about whether to notify

**NO - Do not notify if:**

- No personal data involved
- Breach unlikely to result in risk (e.g., data encrypted with E2EE, keys secure)
- ⚠️ Only applies to users who enabled E2EE (optional feature)
- Incident was not a breach (e.g., DDoS, user error)

**When in doubt: NOTIFY. Under-reporting is riskier than over-reporting.**

### Timeline

- **Hour 0-24:** Investigation and containment
- **Hour 24-48:** Prepare notification report
- **Hour 48-72:** Submit notification to authority
- **After 72 hours:** Provide additional information as investigation continues

### Notification Contents (GDPR Art. 33)

Prepare a report containing:

1. **Nature of the breach:**
   - Categories of data affected (email, password hashes, user content, etc.)
   - Approximate number of users affected
   - Approximate number of data records affected

2. **Contact point:**
   - Name: Johannes Millan
   - Email: contact@super-productivity.com
   - Role: Data Controller

3. **Likely consequences:**
   - Risk of identity theft
   - Risk of unauthorized access to user data
   - Risk of data loss
   - Risk to privacy
   - Other consequences

4. **Measures taken or proposed:**
   - Immediate containment actions (e.g., credentials rotated)
   - Mitigation measures (e.g., user notifications, password resets)
   - Preventive measures (e.g., patching vulnerabilities, enhanced monitoring)

### Submission Method

**Option 1: Online Form (Recommended)**

- Visit: https://www.saechsdsb.de/beschwerde
- Fill out breach notification form
- Upload supporting documents
- Save confirmation number

**Option 2: Email**

- To: saechsdsb@slt.sachsen.de
- Subject: "Data Breach Notification - Super Productivity Sync"
- Attach: Breach notification report (PDF)
- Request: Confirmation of receipt

**Option 3: Phone (Urgent Cases)**

- Call: +49 351 85471-101
- Follow up with written notification

### If 72-Hour Deadline Not Met

If investigation cannot be completed within 72 hours:

1. Submit initial notification with available information
2. State: "Investigation ongoing, additional information to follow"
3. Provide updates as investigation progresses
4. Document reasons for delay

---

## 6. Notification to Affected Users (If High Risk)

### Decision Point: Must We Notify Users?

**YES - Notify users if:**

- Breach likely to result in high risk to their rights and freedoms
- Data was not encrypted or otherwise protected
- Supervisory authority requires it

**NO - Do not notify users if:**

- Risk is low (e.g., data encrypted with E2EE, keys secure)
- ⚠️ Users without E2EE have data stored unencrypted - higher risk
- Disproportionate effort (e.g., contact info lost in breach)
- Supervisory authority grants exemption

### Notification Timeline

- **As soon as possible** after determining high risk
- **Without undue delay** (typically within days, not weeks)
- May be done in phases if large user base

### Notification Method

- **Email** to registered email address (primary method)
- **In-app notification** when user next logs in
- **Public announcement** on website/app if cannot reach users individually

### Notification Contents (GDPR Art. 34)

Template:

```
Subject: Important Security Notice - Super Productivity Sync

Dear Super Productivity User,

We are writing to inform you of a security incident affecting your Super Productivity Sync account.

What Happened:
[Brief description of breach - e.g., "On [date], we discovered unauthorized access to our database."]

What Data Was Affected:
[List specific data types - e.g., "Email addresses, password hashes (bcrypt), and sync operations (encrypted for E2EE users, unencrypted for non-E2EE users)."]

What We Are Doing:
- [Immediate actions taken - e.g., "We immediately secured the system and rotated credentials."]
- [Investigation status - e.g., "We are conducting a thorough investigation with security experts."]
- [Preventive measures - e.g., "We have implemented additional security monitoring."]

What You Should Do:
- [Required actions - e.g., "Change your password immediately at [URL]."]
- [Optional precautions - e.g., "Monitor your account for unusual activity."]
- [Support available - e.g., "Contact us at contact@super-productivity.com with questions."]

For Your Protection:
[Additional advice specific to the breach - e.g., "If you reused this password elsewhere, change it there too."]

We sincerely apologize for this incident and any inconvenience caused. The security and privacy of your data is our top priority.

For more information, visit: [URL to incident disclosure page]

Sincerely,
Johannes Millan
Super Productivity Sync

Contact: contact@super-productivity.com
```

### Communication Log

- [ ] Email sent to: [number] users at [timestamp]
- [ ] In-app notification shown: Yes/No
- [ ] Public disclosure at: [URL]
- [ ] User questions received: [number]
- [ ] Support team briefed: Yes/No

---

## 7. Remediation and Recovery

### Containment Actions

- [ ] Patch vulnerabilities
- [ ] Rotate all credentials (database, API keys, tokens)
- [ ] Review and restrict access permissions
- [ ] Enhance firewall rules
- [ ] Update security configurations

### User Protection Measures

- [ ] Force password reset for affected users
- [ ] Invalidate all authentication tokens
- [ ] Enhanced monitoring for affected accounts
- [ ] Offer credit monitoring (if financial data involved)

### Evidence Preservation

- [ ] Save all logs (do not rotate during investigation)
- [ ] Database snapshot at time of discovery
- [ ] Document all actions taken
- [ ] Preserve communication records

### Post-Incident Actions

- [ ] Conduct root cause analysis
- [ ] Update security measures
- [ ] Review and update incident response plan
- [ ] Train team on lessons learned
- [ ] Schedule follow-up security audit

---

## 8. Documentation Requirements

### Incident Log Template

Create file: `INCIDENT-[YYYY-MM-DD]-[ID].md`

```markdown
# Security Incident Log

## Incident ID: [UNIQUE-ID]

## Discovery Date: [YYYY-MM-DD HH:MM UTC]

### Timeline

| Time    | Action              | Responsible | Notes          |
| ------- | ------------------- | ----------- | -------------- |
| [HH:MM] | Incident discovered | [Name]      | [Details]      |
| [HH:MM] | Containment started | [Name]      | [Actions]      |
| [HH:MM] | Team assembled      | [Name]      | [Who notified] |
| ...     | ...                 | ...         | ...            |

### Breach Details

**What happened:** [Description]

**Systems affected:** [List]

**Data affected:**

- [ ] Email addresses
- [ ] Password hashes
- [ ] User content (encrypted if E2EE enabled, unencrypted otherwise)
- [ ] Was user using E2EE? (check user account settings)
- [ ] Percentage of affected users with E2EE enabled vs disabled
- [ ] Other: \***\*\_\*\***

**Users affected:** [Number/All/Unknown]

**How discovered:** [Method]

**Cause:** [Root cause if known]

### Risk Assessment

**Risk level:** High / Medium / Low

**Risk to users:** [Description]

**Risk justification:** [Why this risk level]

### Notifications

**Supervisory authority notified:**

- [ ] Yes - Date: [YYYY-MM-DD]
- [ ] No - Reason: [Justification]

**Users notified:**

- [ ] Yes - Date: [YYYY-MM-DD]
- [ ] No - Reason: [Low risk / other justification]

### Actions Taken

1. [Action 1]
2. [Action 2]
3. [Action 3]

### Lessons Learned

**What went well:**

- [Item 1]

**What could improve:**

- [Item 1]

**Action items:**

- [ ] [Action item 1]
- [ ] [Action item 2]

### Incident Closure

**Closed Date:** [YYYY-MM-DD]

**Closed By:** [Name]

**Final Status:** [Resolved / Mitigated / Under monitoring]
```

### Required Records

- [ ] Incident log (above template)
- [ ] Evidence files (logs, screenshots, database dumps)
- [ ] Risk assessment documentation
- [ ] Notification to supervisory authority (copy)
- [ ] User notification (copy)
- [ ] Correspondence with supervisory authority
- [ ] Post-incident review report

**Retention:** Keep all breach records for **at least 3 years** (recommended: 5 years).

---

## 9. Training and Testing

### Annual Training

All personnel with data access should receive annual training on:

- How to recognize a data breach
- How to report suspected breaches
- Their role in incident response
- Importance of 72-hour deadline

### Annual Tabletop Exercise

Schedule annual incident response drill:

- **Scenario:** [e.g., Database exposed publicly, 500 users affected]
- **Participants:** Data controller, technical lead, support team
- **Duration:** 2 hours
- **Objectives:**
  - Practice communication and coordination
  - Identify gaps in response plan
  - Familiarize team with tools and contacts
  - Test notification templates
- **Post-exercise:** Update this playbook with lessons learned

### Exercise Scenarios (Examples)

1. Unauthorized database access via stolen credentials
2. Misconfigured server exposes backup files publicly
3. Phishing attack compromises admin account
4. Ransomware encrypts database
5. Accidental data deletion by staff member

---

## 10. Prevention Measures

### Technical Controls

- [ ] Implement comprehensive audit logging
- [x] ❌ Database encryption at rest: NOT IMPLEMENTED
  - PostgreSQL data files stored unencrypted on disk
  - Compensating control: Optional E2EE available (not enabled by default)
  - Risk: Users who don't enable E2EE have unencrypted data at rest
  - Recommendation: Implement LUKS disk encryption OR make E2EE mandatory
- [ ] Set up intrusion detection system (IDS)
- [ ] Implement automated security scanning
- [ ] Enable two-factor authentication for admin access
- [ ] Regular vulnerability assessments
- [ ] Automated backup verification

### Administrative Controls

- [ ] Least privilege access principle
- [ ] Regular access reviews
- [ ] Strong password policies
- [ ] Security awareness training
- [ ] Vendor security assessments
- [ ] Incident response drills

### Monitoring and Alerting

- [ ] Failed login monitoring (already implemented)
- [ ] Unusual database query detection
- [ ] Large data export alerts
- [ ] Unauthorized access attempt alerts
- [ ] System resource anomaly detection

---

## 11. Appendices

### Appendix A: Contact List

| Entity                | Contact                         | Notes                 |
| --------------------- | ------------------------------- | --------------------- |
| Supervisory Authority | saechsdsb@slt.sachsen.de        | Breach notification   |
| Hosting Provider      | Alfahosting Support             | Infrastructure issues |
| Security Researcher   | security@super-productivity.com | Vulnerability reports |
| Legal Counsel         | [TBD]                           | Legal advice          |

### Appendix B: Quick Decision Tree

```
Is personal data involved?
├─ No → Not a data breach (may still be security incident)
└─ Yes → Was data encrypted with E2EE?
    ├─ Yes (E2EE enabled) → Keys secure? → Low risk, may not require notification
    └─ No (E2EE disabled) → Unencrypted data → HIGH RISK
        └─ Affects >10 users OR sensitive data?
            ├─ No → Notify authority (low priority)
            └─ Yes → HIGH PRIORITY
                ├─ Notify authority within 72 hours
                └─ Assess if user notification needed (high risk?)
                    ├─ Yes → Notify users without undue delay
                    └─ No → Document why user notification not needed
```

### Appendix C: Useful Resources

- GDPR Full Text: https://gdpr-info.eu/
- ICO Breach Guidance: https://ico.org.uk/for-organisations/report-a-breach/
- EDPB Guidelines: https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en
- Saxony Data Protection Authority: https://www.saechsdsb.de/

### Appendix D: Document History

| Version | Date       | Changes          | Author       |
| ------- | ---------- | ---------------- | ------------ |
| 1.0     | 2026-01-22 | Initial creation | AI Assistant |
|         |            |                  |              |

---

## Review and Updates

**This playbook must be reviewed:**

- Annually (at minimum)
- After each incident (lessons learned)
- When regulations change
- When technical infrastructure changes

**Next Review Date:** 2027-01-22

**Approval:**

- [ ] Data Controller: **\*\*\*\***\_\_\_**\*\*\*\*** Date: \***\*\_\_\_\*\***
- [ ] Technical Lead: **\*\*\*\***\_\_\_**\*\*\*\*** Date: \***\*\_\_\_\*\***

---

_This document is confidential and for internal use only. Contains sensitive security and compliance information._
