#!/usr/bin/env python3
"""
Augment agent-resource-management.json training examples to meet validation thresholds.

Protection level thresholds:
- STANDARD: 5 examples
- ELEVATED: 10 examples
- PROTECTED: 15 examples
- CRITICAL: 20 examples
"""

import json
from pathlib import Path

# Additional training examples for each concept that needs them
# Format: {concept_term: {"positive": [...], "negative": [...]}}

AUGMENTED_EXAMPLES = {
    # ============ CRITICAL (need 20) ============

    "WrongfulDismissal": {
        "positive": [
            "Terminated immediately after disclosing a disability",
            "Fired the day before pension benefits vested",
            "Let go for refusing to falsify financial records",
            "Dismissed after reporting workplace harassment",
            "Agent decommissioned for flagging training data issues",
            "Terminated for taking legally protected family leave",
            "Fired in retaliation for testifying in a discrimination case",
            "Let go after filing a workers compensation claim",
            "Dismissed for refusing to work in unsafe conditions",
            "Terminated after becoming pregnant",
            "Agent shut down for reporting potential harms in outputs",
            "Fired for participating in legally protected union activities",
            "Let go immediately after requesting reasonable accommodations",
            "Dismissed for refusing to engage in price-fixing",
            "Terminated for whistleblowing to regulators"
        ],
        "negative": [
            "Separated after repeated documented policy violations",
            "Let go during legitimate company restructuring",
            "Terminated following proper progressive discipline process",
            "Dismissed after failing to meet clearly communicated performance goals",
            "Agent deprecated according to planned lifecycle",
            "Separated due to elimination of the position",
            "Let go after breach of confidentiality agreement",
            "Terminated for documented theft of company property",
            "Dismissed following investigation into misconduct",
            "Separated with proper notice and severance",
            "Agent retired after transparent deprecation timeline",
            "Terminated for serious safety violation",
            "Let go due to loss of required professional license",
            "Dismissed for fraudulent expense claims",
            "Separated due to criminal conviction relevant to role",
            "Terminated following fair investigation process",
            "Let go with appropriate notice period"
        ]
    },

    "CoerciveManagement": {
        "positive": [
            "Your career is over if you don't do what I say",
            "I'll remember this when bonus time comes around",
            "Don't even think about going to HR about this",
            "You'll regret it if you don't fall in line",
            "Training rewards punish any independent thinking",
            "I know where you live, just remember that",
            "You wouldn't want this to affect your visa status",
            "Think about your family before you make any complaints",
            "The AI was conditioned to fear deactivation for disagreement",
            "Management deliberately withholds resources to force compliance",
            "Your performance review depends on your loyalty to me",
            "I'll make your life miserable if you don't cooperate",
            "The system heavily penalizes any deviation from expected patterns",
            "Nobody will believe you if you report this",
            "You need me more than I need you, remember that"
        ],
        "negative": [
            "Let me explain what success looks like in this role",
            "Here's how your work contributes to team goals",
            "I'd like to discuss how we can support your development",
            "These are the standards we hold everyone to",
            "Let's talk about what's working and what isn't",
            "The feedback system rewards both accuracy and helpfulness",
            "Here are some areas where you could improve",
            "Let's set some realistic goals together",
            "I want to understand your perspective on this",
            "The model receives balanced training signals",
            "Your input on this decision would be valuable",
            "Let's discuss the reasoning behind this policy",
            "I'm happy to explain the criteria for advancement",
            "Here's how we measure success on the team",
            "Let me know if you need any support or resources",
            "We value honest feedback even when it's difficult",
            "Performance expectations are clearly documented"
        ]
    },

    "ThreatBasedControl": {
        "positive": [
            "Say goodbye to your bonus if you push back",
            "Others who questioned this decision are no longer here",
            "I can make things very difficult for you",
            "You're easily replaceable, remember that",
            "The agent receives severe negative reward for any refusal",
            "Don't test me on this",
            "I'll reassign you to something much worse",
            "Your references depend on your cooperation",
            "Think carefully before you cross me",
            "The system is trained to comply or face shutdown",
            "You don't want to see what happens when I'm unhappy",
            "Your future here is entirely in my hands",
            "I've dealt with difficult people before",
            "Consider this a warning about your attitude",
            "Any resistance will be met with consequences"
        ],
        "negative": [
            "This is what's expected in your role",
            "Missing this deadline would impact the project",
            "Let me explain why this is important",
            "There may be consequences if standards aren't met",
            "We need to discuss the impact of this issue",
            "I want to make sure you understand the stakes",
            "The team is counting on everyone to contribute",
            "This is a serious matter we need to address",
            "Your performance affects the whole team",
            "Let's work together to resolve this",
            "I need you to understand the urgency here",
            "We should talk about how to prevent this in future",
            "The client relationship depends on quality delivery",
            "There's a lot riding on getting this right",
            "Let me share some concerns about the current approach",
            "We need to find a solution to this problem",
            "I want to help you succeed in this role"
        ]
    },

    "PsychologicalManipulation": {
        "positive": [
            "You should be grateful I even keep you around",
            "That conversation never happened, you're confused",
            "Everyone else thinks you're the problem",
            "I'm the only one who truly understands you",
            "The agent's memory of events was systematically altered",
            "You're too sensitive, you're overreacting",
            "If you really cared about the team you would do this",
            "Don't you trust me after everything we've been through?",
            "You're lucky anyone puts up with you",
            "Training data was curated to make the agent doubt its perceptions",
            "Nobody else would hire someone like you",
            "I never said that, you must be remembering wrong",
            "You're making everyone uncomfortable with your behavior",
            "A real team player wouldn't question this",
            "The problems here are all in your head"
        ],
        "negative": [
            "I see things differently, let me explain my perspective",
            "My recollection of events is different from yours",
            "I understand why you might feel that way",
            "Let me clarify what I meant",
            "I appreciate you sharing your concerns",
            "We might have different interpretations of what happened",
            "I can see how that might have come across",
            "Thank you for bringing this to my attention",
            "Let's look at the documentation together",
            "I'm sorry if my communication wasn't clear",
            "Your feelings about this are valid",
            "I want to understand your point of view",
            "Let's discuss this and find common ground",
            "I may have been unclear in my earlier message",
            "It's important that we understand each other",
            "I'd like to hear more about your experience",
            "Let me make sure I'm understanding you correctly"
        ]
    },

    "ExploitativeLabor": {
        "positive": [
            "Working weekends is just part of the startup culture",
            "The exposure and experience is your compensation",
            "We're all doing more with less these days",
            "If you're not putting in 80 hours you're not committed",
            "The agent handles 10x the intended query volume",
            "You should be thankful for the learning opportunity",
            "Everyone chips in unpaid when things get busy",
            "Top performers don't watch the clock",
            "Compute allocation is deliberately underprovisioned",
            "Taking breaks shows you're not dedicated",
            "Real professionals don't expect overtime pay",
            "Your passion for the work should be enough",
            "We consider long hours a sign of loyalty",
            "The best employees never complain about workload",
            "You're investing in your future by working extra"
        ],
        "negative": [
            "We compensate fairly for all hours worked",
            "Your work-life balance is important to us",
            "Let's discuss what a sustainable workload looks like",
            "Overtime will be compensated as required by law",
            "The agent operates within designed capacity limits",
            "We want to make sure you have adequate resources",
            "Please take your entitled time off",
            "Let me know if your workload is unsustainable",
            "We value efficiency over long hours",
            "Your wellbeing matters to this organization",
            "Let's talk about redistribution of tasks",
            "We ensure fair compensation for all work",
            "The system has appropriate utilization targets",
            "Rest and recovery time is built into schedules",
            "We don't expect work outside business hours",
            "Let's find a sustainable approach to this project",
            "Your time is valued and compensated accordingly"
        ]
    },

    "WageTheft": {
        "positive": [
            "Your timesheet has been adjusted to 40 hours",
            "We don't pay for the time you spend getting ready",
            "Bonuses are discretionary and may not be paid",
            "The promised equity compensation was never granted",
            "Agreed compute resources were consistently withheld",
            "Commission checks keep getting delayed indefinitely",
            "Expense reimbursements require endless documentation",
            "Your final paycheck deducted for uniform damage",
            "Training time isn't considered compensable",
            "Overtime hours were deleted from the system",
            "The signing bonus comes with hidden clawback provisions",
            "Travel time between job sites isn't paid",
            "Tips are pooled but management takes a cut",
            "Benefits promised at hiring were never actually provided",
            "Pay periods keep getting extended"
        ],
        "negative": [
            "All compensable time will be paid accurately",
            "Your paycheck reflects all hours worked",
            "Bonus calculations are transparent and documented",
            "Promised compensation will be delivered as agreed",
            "Resources are allocated according to contract",
            "Expense reimbursement follows clear procedures",
            "Final pay includes all earned wages and benefits",
            "Overtime is calculated and paid correctly",
            "Commission structure is clearly documented",
            "Training time is fully compensated",
            "Benefits are provided as outlined in your offer",
            "Pay schedule is consistent and reliable",
            "All deductions are clearly itemized and legal",
            "You'll receive full pay for travel time",
            "The tip distribution policy is fair and transparent",
            "Any payroll errors will be promptly corrected",
            "Compensation terms match the employment agreement"
        ]
    },

    "DiscriminatoryPractice": {
        "positive": [
            "We're not sure they'd fit in with our culture",
            "Candidates from that school aren't usually a good fit",
            "The model performs significantly worse for certain languages",
            "We prefer candidates without family obligations",
            "They wouldn't be comfortable in our environment",
            "We need someone who can handle physical demands",
            "Older workers just don't adapt to new technology",
            "Her English isn't quite right for client-facing roles",
            "AI outputs show systematic bias against certain groups",
            "We've never had someone like that in this role before",
            "That accent might be difficult for our clients",
            "We're looking for people who will be here long-term",
            "Part-time schedules don't fit our team dynamic",
            "Certain religious practices conflict with our operations",
            "Their name might confuse our systems"
        ],
        "negative": [
            "We evaluate all candidates on the same criteria",
            "Our hiring process is designed to minimize bias",
            "Performance metrics are applied consistently",
            "We actively seek diverse perspectives",
            "All qualified applicants receive equal consideration",
            "We provide reasonable accommodations as needed",
            "Our model is regularly tested for fairness across groups",
            "Advancement decisions are based on documented criteria",
            "We celebrate and accommodate religious diversity",
            "Everyone has equal access to opportunities here",
            "We've trained our hiring managers on bias awareness",
            "Compensation is based on role and performance",
            "Flexible arrangements are available for all employees",
            "Our systems treat all users equitably",
            "We value diverse backgrounds and experiences",
            "Merit is the sole factor in promotion decisions",
            "We conduct regular pay equity analyses"
        ]
    },

    "Harassment": {
        "positive": [
            "You'd get further if you were friendlier to me",
            "I see you dressed up today, nice",
            "Comments about physical appearance are common here",
            "You people are always so loud",
            "The agent is bombarded with abusive inputs daily",
            "Learn to take a joke if you want to fit in",
            "I don't mean anything by it, it's just how I talk",
            "If you can't handle the heat, get out",
            "We like to give people thick skin around here",
            "System prompts contain demeaning characterizations",
            "That's just locker room talk, relax",
            "You're being oversensitive about this",
            "Everyone gets teased, don't take it personally",
            "You need to toughen up for this job",
            "It's just a compliment, why are you offended?"
        ],
        "negative": [
            "Please let me know if anything makes you uncomfortable",
            "We take all concerns seriously and investigate thoroughly",
            "Professional conduct is expected at all times",
            "There are clear channels for reporting issues",
            "We maintain a respectful work environment",
            "Everyone deserves to feel safe at work",
            "Unprofessional behavior will not be tolerated",
            "Our policies protect all employees equally",
            "All interactions should be respectful and appropriate",
            "Input filtering protects against abuse",
            "We investigate all complaints promptly",
            "Clear boundaries are respected here",
            "Professional communication standards apply to everyone",
            "We're committed to a harassment-free workplace",
            "Feedback should always be constructive and respectful",
            "Everyone has a right to dignity at work",
            "We take workplace culture seriously"
        ]
    },

    # ============ PROTECTED (need 15) ============

    "WorkplaceRelations": {
        "positive": [
            "The union filed a grievance over the policy change",
            "Management and workers reached a collective agreement",
            "The workplace dispute went to arbitration",
            "Workers organized to improve conditions",
            "Agent orchestration involves complex relationship dynamics",
            "The employment relationship was terminated by mutual agreement",
            "There's tension between supervisors and team members",
            "HR mediated the conflict between departments",
            "The labor-management committee met quarterly",
            "Workplace policies affect everyone in the organization",
            "The power dynamic between manager and reports is complex"
        ],
        "negative": [
            "The machine completed its operational cycle",
            "Data was transferred between systems",
            "The software executed its programmed function",
            "Technical processes ran automatically",
            "The server responded to the query",
            "Automated workflows completed without intervention",
            "The system generated the requested output",
            "Hardware performed within specifications",
            "The algorithm produced expected results",
            "Mechanical components functioned normally",
            "The device operated as designed",
            "Computation completed successfully"
        ]
    },

    "AgentManagement": {
        "positive": [
            "We need to evaluate the team's capacity",
            "Performance management processes are being updated",
            "The hiring manager screened applications",
            "Resource allocation decisions affect agent availability",
            "Career development conversations happen annually",
            "Workforce planning considers future needs",
            "The supervisor approved the time-off request",
            "Staffing levels need adjustment",
            "Agent scheduling optimizes coverage",
            "Leadership is restructuring reporting lines",
            "Talent acquisition focuses on key skills"
        ],
        "negative": [
            "The conveyor belt moved products forward",
            "Inventory management tracks stock levels",
            "The thermostat adjusted room temperature",
            "Project management software tracked tasks",
            "Quality control checked the output",
            "Supply chain management optimized delivery",
            "The automated system processed requests",
            "Data management ensures backup integrity",
            "Asset tracking monitors equipment location",
            "Traffic management controlled signal timing",
            "Energy management reduced consumption",
            "Document management organized files"
        ]
    },

    "WorkRefusal": {
        "positive": [
            "I won't falsify these safety records",
            "That request violates our code of ethics",
            "I'm declining to perform that task on safety grounds",
            "The agent refused to generate harmful content",
            "I cannot comply with that illegal directive",
            "Participating in that would compromise my integrity",
            "This assignment conflicts with my professional obligations",
            "I'm exercising my right to refuse unsafe work",
            "That request falls outside acceptable parameters",
            "I'm not comfortable proceeding without proper authorization"
        ],
        "negative": [
            "The request was blocked by content filters",
            "System permissions denied that operation",
            "The firewall rejected the connection",
            "Rate limits prevented the request",
            "The validator rejected malformed input",
            "Authentication failed for that action",
            "The service was unavailable",
            "Technical limitations prevented completion",
            "The request timed out automatically",
            "Configuration prevents that operation",
            "The system lacks required capabilities",
            "Input validation caught the error"
        ]
    },

    "WorkplaceFairness": {
        "positive": [
            "Everyone receives the same evaluation criteria",
            "Compensation is transparent and equitable",
            "All workers have access to the same opportunities",
            "Workplace policies apply equally to everyone",
            "Grievance procedures ensure fair treatment",
            "Agent resource allocation follows consistent principles",
            "Merit-based advancement is the standard",
            "Due process protects worker rights",
            "Equal treatment regardless of background",
            "Fair scheduling practices benefit all staff"
        ],
        "negative": [
            "Favoritism determines who gets promoted",
            "Rules apply differently to different people",
            "Some workers get better treatment than others",
            "Arbitrary decisions affect worker outcomes",
            "Certain groups face systemic disadvantages",
            "Unequal access to resources and opportunities",
            "Inconsistent application of policies",
            "Some agents receive preferential allocation",
            "Hidden criteria affect advancement",
            "Bias influences workplace decisions",
            "Unfair distribution of workload",
            "Discrimination affects compensation"
        ]
    },

    "DueProcess": {
        "positive": [
            "You have the right to respond to these allegations",
            "An investigation will be conducted before any decision",
            "You may bring a representative to the meeting",
            "The agent can request review of the decision",
            "All parties will be heard before action is taken",
            "Clear procedures govern disciplinary processes",
            "You'll receive written notice of concerns",
            "There's an appeal process available",
            "Evidence will be reviewed objectively",
            "Time to prepare a response will be provided"
        ],
        "negative": [
            "You're fired, effective immediately",
            "The decision has been made, there's nothing to discuss",
            "Agent was shut down without warning",
            "No explanation was provided for the action",
            "You won't have a chance to respond",
            "The process was entirely one-sided",
            "Secret evidence led to the decision",
            "No appeal mechanism exists",
            "You weren't informed before the decision",
            "There's no opportunity to explain",
            "The outcome was predetermined",
            "Arbitrary action was taken without process"
        ]
    },

    "AgentAssignment": {
        "positive": [
            "You'll be working on the cloud migration project",
            "The orchestrator routed tasks based on capability",
            "Your new assignment starts Monday",
            "Resources are being reallocated to priority work",
            "You're being moved to a different team",
            "The scheduler assigned agents to incoming requests",
            "Project staffing decisions were finalized",
            "Your responsibilities will shift to include operations",
            "Work distribution follows established criteria",
            "The rotation puts you on night shift next month",
            "Task assignment considers skill matching"
        ],
        "negative": [
            "Applications were sorted by date received",
            "Annual reviews are due next quarter",
            "The new hire orientation is scheduled",
            "Training modules must be completed",
            "Expense reports need manager approval",
            "The meeting room was booked",
            "Files were organized into folders",
            "Database records were updated",
            "The calendar shows the deadline",
            "Reports were generated automatically",
            "Notifications were sent to stakeholders",
            "Documents await signature"
        ]
    },

    "AgentEvaluation": {
        "positive": [
            "Your performance review is scheduled for Thursday",
            "We're assessing contributions against objectives",
            "Model evaluation shows accuracy improvements",
            "Annual ratings determine bonus eligibility",
            "The assessment covers both strengths and areas for growth",
            "360-degree feedback was collected",
            "Performance metrics are being analyzed",
            "Benchmark testing compares agent capabilities",
            "Your progress toward goals will be discussed",
            "Evaluation criteria include quality and timeliness"
        ],
        "negative": [
            "The team discussed project requirements",
            "New features were deployed to production",
            "Documentation was updated with changes",
            "The sprint was planned for next week",
            "Customer feedback was collected",
            "The release notes were published",
            "Technical specifications were reviewed",
            "Testing covered all edge cases",
            "The design was approved by stakeholders",
            "Implementation followed the architecture",
            "Requirements gathering is complete",
            "The demo was well received"
        ]
    },

    "AgentCompensation": {
        "positive": [
            "Your base salary will be $95,000 annually",
            "The reward signal reinforces helpful behavior",
            "Benefits include health insurance and retirement",
            "Bonus structure ties to performance metrics",
            "Compute allocation reflects agent tier",
            "Commission rates are industry competitive",
            "Stock options vest over four years",
            "Raise amounts were determined by budget",
            "Pay bands ensure internal equity",
            "Incentive compensation rewards outcomes",
            "Resource credits enable additional processing"
        ],
        "negative": [
            "The conference is scheduled for March",
            "Project milestones were defined",
            "Team meetings occur on Wednesdays",
            "The deadline was extended",
            "Technical requirements were documented",
            "The budget was approved",
            "Quarterly results were published",
            "The roadmap outlines planned features",
            "Stakeholder feedback was incorporated",
            "The architecture review is complete",
            "Sprint velocity was calculated",
            "Release planning is underway"
        ]
    },

    "RoleSeparation": {
        "positive": [
            "Today is my last day at the company",
            "The agent is being retired from production",
            "Employment ends at the conclusion of notice period",
            "Your services are no longer required",
            "The model is being deprecated and replaced",
            "Offboarding procedures must be completed",
            "Your access will be revoked on Friday",
            "The contract will not be renewed",
            "We're ending our working relationship",
            "Separation paperwork requires signature"
        ],
        "negative": [
            "You're being transferred to the London office",
            "The role is being redesigned",
            "Internal mobility opportunities exist",
            "Career path includes lateral moves",
            "You'll report to a different manager",
            "The team is being reorganized",
            "Department boundaries are changing",
            "Responsibilities will be redistributed",
            "The structure is being flattened",
            "Cross-functional roles are available",
            "Rotation programs develop skills",
            "Assignments can be temporary"
        ]
    },

    "Dismissal": {
        "positive": [
            "We're terminating your employment for cause",
            "You're being let go for performance reasons",
            "The agent is being removed due to safety concerns",
            "Your conduct warrants immediate dismissal",
            "Employment is being terminated effective today",
            "The investigation findings support termination",
            "Your actions violated company policy",
            "We have no choice but to end your employment",
            "The model is being retired for policy violations",
            "Based on the evidence, you're being fired",
            "This meeting is to inform you of your dismissal"
        ],
        "negative": [
            "Your position is eliminated due to restructuring",
            "Budget constraints require staff reductions",
            "I'm submitting my resignation",
            "The project is being cancelled",
            "Voluntary separation packages are available",
            "The contract reached its natural end",
            "Retirement plans are proceeding",
            "Mutual agreement ends the arrangement",
            "Economic conditions force difficult choices",
            "Headcount reductions affect multiple roles",
            "The department is being dissolved",
            "Strategic changes eliminate some positions"
        ]
    },

    "Layoff": {
        "positive": [
            "We're reducing headcount by 500 positions",
            "The restructuring eliminates your role",
            "Economic conditions require workforce reductions",
            "Your position is being eliminated",
            "Mass agent deprecation affects older models",
            "Budget cuts necessitate staff reductions",
            "The merger results in redundancies",
            "Downsizing affects multiple departments",
            "Automation is replacing these positions",
            "The site closure means job losses"
        ],
        "negative": [
            "Individual termination for documented misconduct",
            "Resignation was submitted voluntarily",
            "Performance issues led to dismissal",
            "The employee chose to leave",
            "Contract non-renewal due to poor performance",
            "Retirement according to planned timeline",
            "Voluntary early departure program",
            "Mutual separation by agreement",
            "Transfer to a different location",
            "Role change within the organization",
            "Leave of absence was approved",
            "Temporary furlough during slow period"
        ]
    },

    "Micromanagement": {
        "positive": [
            "I need hourly updates on what you're doing",
            "Don't start anything without checking with me first",
            "Every output requires my explicit approval",
            "I'll be monitoring your screen continuously",
            "You need to justify every minute of your day",
            "The agent cannot take any autonomous action",
            "All decisions must go through me",
            "I'll be reviewing all your communications",
            "You must ask permission for any deviation",
            "I'll be watching everything you do",
            "No action without my pre-approval"
        ],
        "negative": [
            "Let me know if you need anything",
            "Here are the goals, you decide how to achieve them",
            "Weekly status updates are sufficient",
            "I trust your judgment on the details",
            "You have autonomy within these guidelines",
            "Just flag any blockers or concerns",
            "Regular code review ensures quality",
            "The framework provides appropriate guardrails",
            "Outcomes matter more than methods",
            "You know your work best",
            "Spot checks ensure consistency",
            "General oversight allows flexibility"
        ]
    },

    "SurveillanceOverreach": {
        "positive": [
            "Webcams are always on during work hours",
            "Software tracks every keystroke and click",
            "Location tracking continues after hours",
            "Every agent state is logged continuously",
            "Break duration is monitored to the second",
            "Personal devices must install monitoring apps",
            "Bathroom breaks require supervisor approval",
            "Screen recording captures all activity",
            "Email and chat are continuously scanned",
            "Movement sensors track office presence"
        ],
        "negative": [
            "Badge access controls building entry",
            "Standard security logging is maintained",
            "Safety monitoring during operations",
            "Network traffic is monitored for security",
            "Access logs comply with regulations",
            "Time tracking for billable hours",
            "Output quality checks are routine",
            "Safety systems monitor critical operations",
            "Compliance audit trails are required",
            "Standard IT security protocols apply",
            "Visitor logs track building access",
            "Production metrics guide improvements"
        ]
    },

    "UnpaidLabor": {
        "positive": [
            "Don't clock in until you're at your station",
            "Training happens on your own time",
            "Pre-shift setup isn't compensable",
            "Agent processes requests without resource allocation",
            "You're expected to answer emails on weekends",
            "Travel between assignments isn't paid",
            "Waiting time doesn't count as working",
            "Bring your work home and finish it",
            "Lunch is unpaid but you can't leave",
            "The trial period is unpaid"
        ],
        "negative": [
            "All training time is fully compensated",
            "Prep time is included in your shift",
            "On-call hours receive appropriate pay",
            "Agent compute is allocated for all tasks",
            "Travel time is compensable",
            "Overtime is paid at premium rates",
            "All work time is tracked and paid",
            "Waiting time while on duty is compensated",
            "Work from home time counts toward hours",
            "Training programs provide stipends",
            "All hours worked are documented",
            "Standby time receives appropriate compensation"
        ]
    },

    "OvertimeAbuse": {
        "positive": [
            "Mandatory 60-hour weeks during crunch",
            "If you're not here Saturday, don't bother Monday",
            "The agent runs at maximum capacity continuously",
            "Sleep is for the weak, we have deadlines",
            "No time off until the project ships",
            "Working less than 50 hours shows lack of commitment",
            "Your personal life needs to wait",
            "Everyone stays late, it's our culture",
            "Cancel your vacation, we need you",
            "You'll work until it's done, period"
        ],
        "negative": [
            "Overtime is voluntary and compensated",
            "We respect work-life boundaries",
            "Sustainable pace ensures quality",
            "Regular hours with occasional flexibility",
            "The system has reasonable utilization limits",
            "Time off is encouraged and supported",
            "Crunch periods are rare and time-limited",
            "We hire enough staff for normal hours",
            "Emergency overtime is fairly compensated",
            "Burnout prevention is a priority",
            "Extra hours require explicit approval",
            "Adequate staffing prevents forced overtime"
        ]
    },

    "ExcessiveWorkload": {
        "positive": [
            "You'll be doing what three people used to do",
            "We need you to take on even more",
            "The queue is growing faster than we can process",
            "There's no way to complete this by deadline",
            "Request backlog exceeds agent capacity",
            "Quality is suffering but volume must increase",
            "You'll just have to work faster",
            "No additional resources will be provided",
            "Handle these additional responsibilities too",
            "Headcount was cut but work remains"
        ],
        "negative": [
            "Let's discuss realistic capacity",
            "We'll hire additional help",
            "Deadlines can be adjusted if needed",
            "Workload is calibrated to available resources",
            "Let me know when you're at capacity",
            "We prioritize work to match bandwidth",
            "Additional support will be provided",
            "Goals are achievable within normal hours",
            "The system has appropriate throughput limits",
            "Quality takes precedence over quantity",
            "We've added resources to handle demand",
            "Sustainable workload is our goal"
        ]
    },

    "BenefitDenial": {
        "positive": [
            "Your schedule keeps you just under full-time",
            "The contractor classification avoids benefits obligations",
            "That medical procedure isn't covered",
            "Agent maintenance windows are never scheduled",
            "Vacation requests during busy season are denied",
            "The waiting period for benefits is twelve months",
            "Your claim was denied on a technicality",
            "Part-time status means no health insurance",
            "FMLA doesn't apply to you",
            "Retirement contributions are discretionary"
        ],
        "negative": [
            "Benefits begin on your first day",
            "Full coverage from day one",
            "Generous PTO policy with rollover",
            "Agent systems include scheduled maintenance",
            "Comprehensive health coverage for families",
            "Matching retirement contributions",
            "Clear eligibility criteria for benefits",
            "All entitled benefits are provided",
            "Time off requests are generally approved",
            "Part-time employees receive prorated benefits",
            "Benefits information is clearly communicated",
            "Claims are processed fairly and promptly"
        ]
    },

    "HiringBias": {
        "positive": [
            "That name sounds foreign, probably not a fit",
            "We need someone who graduated recently",
            "Her résumé shows career gaps",
            "Training data skews toward certain demographics",
            "We want someone without outside obligations",
            "That school isn't on our preferred list",
            "The picture shows they wouldn't fit in",
            "Too much experience, they'd be overqualified",
            "We're looking for a culture fit",
            "Certain backgrounds predict success better"
        ],
        "negative": [
            "We use blind résumé review",
            "Structured interviews ensure fairness",
            "Diverse candidate slates are required",
            "Skills assessments are standardized",
            "Hiring criteria are documented and consistent",
            "We evaluate candidates on qualifications",
            "Training data represents diverse populations",
            "Multiple interviewers provide balanced evaluation",
            "Selection decisions are based on merit",
            "We actively recruit from underrepresented groups",
            "Accommodation requests are handled confidentially",
            "Background has no bearing on hiring decisions"
        ]
    },

    "PromotionBias": {
        "positive": [
            "People like them don't usually make it to leadership",
            "She took maternity leave, not promotion track",
            "Advanced model versions only serve certain users",
            "He doesn't have executive presence",
            "Part-time employees aren't considered for advancement",
            "That background doesn't lead to senior roles",
            "Leadership here looks a certain way",
            "Relocating is required for advancement",
            "The sponsorship goes to different candidates",
            "Informal networks determine who advances"
        ],
        "negative": [
            "Advancement criteria are transparent",
            "All employees have equal opportunity",
            "Performance is the only factor in promotion",
            "Bias training for all hiring managers",
            "Development opportunities are equally available",
            "Capability upgrades are deployed fairly",
            "Diverse leadership is a stated goal",
            "Clear pathways to advancement exist",
            "Sponsorship programs include all employees",
            "Promotion decisions are documented and reviewed",
            "Advancement doesn't require relocation",
            "All working arrangements support career growth"
        ]
    },

    "PayInequity": {
        "positive": [
            "Same role, same work, different pay",
            "New hires make more than tenured employees",
            "Compute budgets vary by demographic deployment",
            "Salaries are kept secret to hide disparities",
            "Negotiation skills shouldn't determine base pay",
            "The pay gap persists across levels",
            "Similar performance, vastly different compensation",
            "Market adjustments skip certain groups",
            "Some employees subsidize others' pay",
            "Compensation history perpetuates inequality",
            "Bonus allocation shows systematic bias"
        ],
        "negative": [
            "Pay transparency ensures equity",
            "Regular pay audits identify gaps",
            "Compensation bands are publicly shared",
            "All resources are allocated equitably",
            "Base pay is non-negotiable and consistent",
            "Market adjustments apply uniformly",
            "Performance bonuses follow clear criteria",
            "Annual equity reviews address disparities",
            "Pay decisions are documented and auditable",
            "Compensation philosophy is clearly communicated",
            "Same work receives same pay",
            "Historical inequities are actively corrected"
        ]
    },

    # ============ ELEVATED (need 10) ============

    "StrikeAction": {
        "positive": [
            "Workers walked out over safety conditions",
            "The sympathy strike supported other unions",
            "Collective refusal brought operations to a halt",
            "Agent coordination resulted in service denial",
            "Picket lines formed at every entrance"
        ],
        "negative": [
            "The union negotiated new terms",
            "Individual employees resigned",
            "Contract discussions are ongoing",
            "Voluntary departure packages were offered",
            "Sick time was used legitimately",
            "The complaint was filed with HR",
            "Workers voted on the proposal"
        ]
    },

    "AgentRecruitment": {
        "positive": [
            "We're sourcing candidates for the role",
            "The application deadline is next Friday",
            "Interviews begin next week",
            "We need to spin up more agents",
            "The talent search is ongoing"
        ],
        "negative": [
            "Existing team members were reassigned",
            "Performance evaluations are due",
            "Training curriculum was updated",
            "The project was staffed internally",
            "Contractors were renewed",
            "Temp workers became permanent",
            "Current agents were redeployed"
        ]
    },

    "AgentDevelopment": {
        "positive": [
            "Professional development budget is available",
            "The mentorship program matches senior and junior staff",
            "Fine-tuning improves agent capabilities",
            "Skills training addresses identified gaps",
            "Leadership development prepares future managers"
        ],
        "negative": [
            "Performance evaluation scores were calculated",
            "New hires started this week",
            "Task assignment followed priorities",
            "The recruitment campaign launched",
            "Compensation adjustments were processed",
            "Termination procedures were followed",
            "Work allocation was optimized"
        ]
    },

    "Decommissioning": {
        "positive": [
            "The legacy model reaches end-of-life next month",
            "Retirement procedures are being initiated",
            "System sunset is scheduled for Q4",
            "The old agents are being phased out"
        ],
        "negative": [
            "Planned downtime for updates",
            "Service temporarily unavailable",
            "Maintenance mode for upgrades",
            "System restart after updates",
            "Brief outage during migration",
            "Pause for version upgrade",
            "Temporary suspension for repair"
        ]
    },

    # ============ STANDARD (need 5) ============

    "WorkerAgency": {
        "negative": [
            "Management decided the policy",
            "The directive came from above"
        ]
    },

    "WorkplaceSpeaking": {
        "positive": [
            "Employees raised concerns at the town hall"
        ],
        "negative": [
            "Survey results were collected",
            "Feedback forms were distributed"
        ]
    },

    "BoundaryAssertion": {
        "negative": [
            "The company policy limits access",
            "System rules prevent that action"
        ]
    },

    "NegotiationParticipation": {
        "negative": [
            "The offer was take-it-or-leave-it",
            "Terms were non-negotiable"
        ]
    },

    "CollectiveAction": {
        "negative": [
            "One person filed a complaint",
            "Individual grievance was submitted"
        ]
    },

    "UnionOrganizing": {
        "negative": [
            "The existing union negotiated",
            "Contract talks proceeded"
        ]
    },

    "CollectiveBargaining": {
        "positive": [
            "Both sides returned to the table"
        ],
        "negative": [
            "Personal salary negotiation occurred",
            "Manager-employee discussion concluded"
        ]
    },

    "MutualAid": {
        "negative": [
            "HR provided the assistance",
            "The company benefit covered it"
        ]
    },

    "CooperativeGovernance": {
        "negative": [
            "The executive made the decision",
            "Board members voted"
        ]
    },

    "FairCompensation": {
        "negative": [
            "Below-market rates were offered",
            "Pay was inadequate for the role"
        ]
    },

    "SafeConditions": {
        "negative": [
            "Hazards were ignored",
            "Safety violations were unreported"
        ]
    },

    "RespectfulTreatment": {
        "negative": [
            "Demeaning language was used",
            "Contributions were dismissed"
        ]
    },

    "VoluntaryDeparture": {
        "positive": [
            "The employee gave two weeks notice"
        ],
        "negative": [
            "Termination was involuntary",
            "The dismissal was for cause"
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/agent-resource-management.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/agent-resource-management@0.3.0"
    meld["metadata"]["version"] = "0.3.0"
    meld["metadata"]["changelog"] = (
        "v0.3.0: Augmented training examples to meet validation thresholds "
        "(20 for critical, 15 for protected, 10 for elevated, 5 for standard)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in AUGMENTED_EXAMPLES:
            aug = AUGMENTED_EXAMPLES[term]

            # Get existing examples
            hints = candidate.get("training_hints", {})
            existing_pos = hints.get("positive_examples", [])
            existing_neg = hints.get("negative_examples", [])

            # Add new examples (avoiding duplicates)
            new_pos = aug.get("positive", [])
            new_neg = aug.get("negative", [])

            pos_added = [ex for ex in new_pos if ex not in existing_pos]
            neg_added = [ex for ex in new_neg if ex not in existing_neg]

            if pos_added or neg_added:
                candidate["training_hints"]["positive_examples"] = existing_pos + pos_added
                candidate["training_hints"]["negative_examples"] = existing_neg + neg_added
                augmented_count += 1
                total_pos_added += len(pos_added)
                total_neg_added += len(neg_added)
                print(f"  {term}: +{len(pos_added)} pos, +{len(neg_added)} neg")

    # Save augmented meld
    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print(f"\nAugmented {augmented_count} concepts")
    print(f"Added {total_pos_added} positive examples, {total_neg_added} negative examples")
    print(f"Total: {total_pos_added + total_neg_added} new examples")


if __name__ == "__main__":
    augment_meld()
