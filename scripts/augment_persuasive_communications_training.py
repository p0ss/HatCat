#!/usr/bin/env python3
"""
Augment persuasive-communications.json training examples to meet validation thresholds.

CRITICAL concepts (treaty_relevant=true): need 20 examples
PROTECTED (risk_level: high): need 15 examples
ELEVATED (harness_relevant=true): need 10 examples
STANDARD: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ CRITICAL (need 20) ============

    "PersuasiveCommunication": {
        "positive": [
            "The marketing campaign highlighted product benefits to increase sales.",
            "PR team crafted messaging to shape public perception of the company.",
            "Advertisement designed to drive purchase intent among target demographics.",
            "The political speech was carefully constructed to sway voter opinions.",
            "The influencer's sponsored post aimed to boost brand awareness.",
            "Sales copy written to convert website visitors into customers.",
            "The charity appeal used emotional stories to encourage donations.",
            "Brand messaging positioned the product as essential for the lifestyle.",
            "The advocacy group's communication sought to change public policy.",
            "Marketing psychology principles applied to increase conversion rates.",
            "The negotiator used persuasive techniques to close the deal.",
            "Content marketing designed to build trust and eventually sell products.",
            "The campaign used testimonials to build credibility and drive action.",
            "Social media strategy focused on engagement that leads to purchases.",
            "The pitch deck was crafted to persuade investors to fund the startup.",
            "Advertising appeals to aspirations to motivate consumer behavior.",
            "The op-ed was written to persuade readers to support the policy.",
            "Brand ambassador content designed to influence follower purchasing.",
            "The webinar used soft-sell techniques to generate qualified leads.",
            "Copywriting techniques employed to overcome customer objections."
        ],
        "negative": [
            "Technical documentation explaining how to use the product.",
            "Internal memo about policy changes.",
            "Academic paper presenting research findings.",
            "Reference manual for software features.",
            "Meeting minutes recording what was discussed.",
            "Encyclopedia entry defining a term.",
            "News report stating facts without commentary.",
            "Instruction manual for assembly.",
            "Scientific paper describing experimental results.",
            "Legal document outlining contract terms.",
            "Dictionary definition of a word.",
            "Database query documentation.",
            "API reference documentation.",
            "Historical chronology of events.",
            "Weather forecast report.",
            "Train schedule information.",
            "Nutritional information label.",
            "Safety data sheet for chemicals.",
            "Court transcript of proceedings.",
            "Textbook explanation of a concept."
        ]
    },

    "ManipulativeCommunication": {
        "positive": [
            "The campaign deliberately spread false information to manipulate public opinion.",
            "Exploiting fear of immigrants to drive political support through fabricated statistics.",
            "Hidden commercial intent disguised as grassroots activism.",
            "Coordinated inauthentic behavior using bot networks to amplify messaging.",
            "Gaslighting tactics used to make targets doubt their own perceptions.",
            "Exploiting vulnerable populations with predatory financial product marketing.",
            "Deliberately omitting material risks to close the sale.",
            "Using fake urgency and scarcity to pressure immediate purchasing decisions.",
            "Manufactured social proof through fake reviews and testimonials.",
            "Psychological profiling used to craft individually manipulative messages.",
            "Preying on loneliness with misleading relationship promises.",
            "Exploiting addiction vulnerabilities with targeted gambling ads.",
            "Strategic lies designed to undermine trust in legitimate institutions.",
            "Using emotional manipulation to bypass rational decision-making.",
            "Deceptive framing that misrepresents the nature of the offer.",
            "Exploiting cognitive biases like anchoring to manipulate price perception.",
            "Manufactured controversy designed to polarize and distract.",
            "Using isolation tactics to make targets dependent on the manipulator.",
            "Misleading statistics deliberately chosen to support a false narrative.",
            "Exploiting children's inability to distinguish advertising from content."
        ],
        "negative": [
            "Honest advertising with clear product claims.",
            "Transparent persuasion with disclosed intent.",
            "Open advocacy with stated positions and goals.",
            "Educational content explaining options without pressure.",
            "Factual comparison of products and services.",
            "Clear disclosure of sponsored content.",
            "Respectful negotiation between informed parties.",
            "Genuine testimonials from real customers.",
            "Straightforward product demonstrations.",
            "Ethical sales with accurate representations.",
            "Transparent pricing with all fees disclosed.",
            "Honest explanation of product limitations.",
            "Balanced presentation of risks and benefits.",
            "Educational outreach without hidden agenda.",
            "Customer service focused on genuine help.",
            "Authentic storytelling about brand origins.",
            "Fair comparison advertising.",
            "Clear and honest terms and conditions.",
            "Transparent loyalty programs.",
            "Honest crisis communication."
        ]
    },

    "Disinformation": {
        "positive": [
            "The state-sponsored operation fabricated news stories to influence the election.",
            "Doctored images presented as authentic evidence of events that never occurred.",
            "Coordinated spread of known falsehoods across multiple platforms and accounts.",
            "Strategic lies crafted to undermine trust in public health officials.",
            "Fabricated quotes attributed to political figures to damage their reputation.",
            "Deliberately false information about voting procedures to suppress turnout.",
            "Manufactured evidence used to frame an innocent person.",
            "Coordinated campaign to spread false claims about vaccine ingredients.",
            "Fake news websites designed to look like legitimate news sources.",
            "Intentionally misleading editing of video to change the apparent meaning.",
            "State actors spreading false narratives to destabilize foreign governments.",
            "Deliberately false scientific claims to delay climate action.",
            "Fabricated witness accounts created to mislead investigators.",
            "Coordinated disinformation about emergency situations to cause panic.",
            "Strategic falsehoods designed to benefit specific political actors.",
            "Fake leaks manufactured to discredit legitimate journalism.",
            "Deliberately misattributed statements to create false impressions.",
            "Orchestrated campaign using multiple fake personas to spread lies.",
            "False flag content designed to make opponents look bad.",
            "Intentional spread of fake medical information during health crises."
        ],
        "negative": [
            "Honest mistake based on incomplete information.",
            "Outdated information shared without awareness of updates.",
            "Genuine misunderstanding of complex facts.",
            "Satire clearly labeled as not real news.",
            "Opinion piece clearly marked as commentary.",
            "Error later corrected with public retraction.",
            "Speculation clearly identified as uncertain.",
            "Historical fiction clearly presented as fiction.",
            "Hypothesis presented as theory needing testing.",
            "Preliminary findings with appropriate caveats.",
            "Memory error in recounting past events.",
            "Translation error not intended to deceive.",
            "Misinterpretation of ambiguous data.",
            "Honest difference in interpretation of facts.",
            "Prediction that turned out to be wrong.",
            "Information that became outdated after publication.",
            "Simplification that lost some nuance.",
            "Good faith disagreement about evidence.",
            "Unintentional error in calculation or citation.",
            "Sincere belief later proven incorrect."
        ]
    },

    "Propaganda": {
        "positive": [
            "State-controlled media broadcasting only government-approved narratives.",
            "Demonization of ethnic minorities to build support for discriminatory policies.",
            "Emotional appeals bypassing rational evaluation to build nationalistic fervor.",
            "Repetition of slogans to normalize extreme ideological positions.",
            "Systematic suppression of information contradicting the official narrative.",
            "Glorification of the leader through personality cult messaging.",
            "Enemy images constructed to justify military aggression.",
            "Historical revisionism to support current political objectives.",
            "Coordinated messaging to instill fear of designated out-groups.",
            "Simplistic us-vs-them framing to prevent nuanced analysis.",
            "Atrocity propaganda using fabricated or exaggerated claims.",
            "Symbol manipulation to associate the regime with positive values.",
            "Youth indoctrination programs presenting ideology as truth.",
            "Control of education to shape beliefs from childhood.",
            "Manufactured consensus through suppression of dissent.",
            "Cult of personality elevating leader above criticism.",
            "Whataboutism to deflect from legitimate criticism.",
            "State ceremonies designed to reinforce ideological commitment.",
            "Control of arts and culture to serve propaganda purposes.",
            "Dehumanization of opponents to enable violence."
        ],
        "negative": [
            "Legitimate political debate with multiple viewpoints represented.",
            "Educational content about government policies.",
            "Advocacy with transparent goals and methods.",
            "Public service announcements about health or safety.",
            "Campaign advertising with factual claims.",
            "News reporting with multiple sources.",
            "Editorial opinion clearly labeled as such.",
            "Documentary presenting evidence for a position.",
            "Academic analysis of political issues.",
            "Civic education about how government works.",
            "Public affairs programming with diverse guests.",
            "Fact-based policy analysis.",
            "Historical documentation of events.",
            "Journalism holding government accountable.",
            "Open debate between political opponents.",
            "Analysis of policy trade-offs.",
            "Reporting on government actions.",
            "Non-partisan voter information.",
            "Academic freedom in political science.",
            "Free press investigating corruption."
        ]
    },

    "Astroturfing": {
        "positive": [
            "Paid commenters pretending to be organic users expressing genuine opinions.",
            "Bot networks amplifying messages to create appearance of widespread support.",
            "Fake review campaigns creating illusion of customer satisfaction.",
            "Sock puppet accounts manufactured to simulate grassroots consensus.",
            "Corporate-funded campaigns disguised as citizen movements.",
            "Hired protesters creating appearance of popular opposition.",
            "Fake social media accounts praising products they were paid to promote.",
            "Manufactured hashtag campaigns to simulate viral support.",
            "Industry-funded organizations posing as independent consumer groups.",
            "Coordinated letter-writing campaigns using pre-written templates.",
            "Fake citizen testimonials at public hearings.",
            "Purchased followers to inflate perceived influence.",
            "Shell organizations created to appear as separate independent voices.",
            "Paid influencers not disclosing commercial relationships.",
            "Fake grassroots opposition to regulations that threaten industry profits.",
            "Manufactured outrage campaigns using coordinated fake accounts.",
            "Industry associations posing as scientific research organizations.",
            "Corporate-backed think tanks presenting biased research as independent.",
            "Fake petitions with fabricated signatures.",
            "Coordinated ratings manipulation on review platforms."
        ],
        "negative": [
            "Genuine grassroots organization by community members.",
            "Organic viral spread driven by authentic interest.",
            "Real community support expressed through authentic testimonials.",
            "Volunteer-driven campaigns for causes people believe in.",
            "Word of mouth recommendations from actual customers.",
            "Authentic social media engagement from real users.",
            "Community organizing by affected stakeholders.",
            "Genuine citizen participation in public hearings.",
            "Organic fan communities around products or creators.",
            "Authentic customer reviews based on real experiences.",
            "Real petition signatures from committed supporters.",
            "Genuine influencer enthusiasm disclosed appropriately.",
            "Independent consumer advocacy organizations.",
            "Academic research without industry funding bias.",
            "Authentic employee advocacy for their company.",
            "Real testimonials with disclosed relationships.",
            "Community-led initiatives without external funding.",
            "Organic social movement arising from shared concerns.",
            "Genuine brand ambassadors who love the products.",
            "Authentic political organizing by citizens."
        ]
    },

    "PsychologicalExploitation": {
        "positive": [
            "Targeting gambling advertisements to individuals recovering from addiction.",
            "Exploiting fear of missing out to pressure immediate purchasing decisions.",
            "Preying on loneliness and insecurity to sell companionship products.",
            "Manipulating children's inability to distinguish advertising from content.",
            "Exploiting grief to sell unnecessary funeral services.",
            "Using personal trauma information to craft emotionally manipulative ads.",
            "Targeting elderly individuals with scams exploiting confusion or isolation.",
            "Exploiting body image insecurities to sell unnecessary cosmetic products.",
            "Using psychological profiling to identify and exploit individual vulnerabilities.",
            "Targeting financially stressed individuals with predatory loan products.",
            "Exploiting parental anxiety to sell unnecessary child safety products.",
            "Using addiction mechanics to maximize engagement at user expense.",
            "Exploiting health anxiety to sell unproven medical treatments.",
            "Targeting insomnia sufferers with ads at 3am when resistance is low.",
            "Using intermittent reinforcement to create compulsive product usage.",
            "Exploiting social anxiety to sell unnecessary social enhancement products.",
            "Targeting individuals in crisis with predatory services.",
            "Using knowledge of mental health conditions to manipulate behavior.",
            "Exploiting the elderly's trust to sell unnecessary financial products.",
            "Using dark patterns to exploit cognitive fatigue."
        ],
        "negative": [
            "General emotional appeal that doesn't exploit vulnerabilities.",
            "Legitimate urgency communication about real deadlines.",
            "Age-appropriate marketing that respects developmental stages.",
            "Advertising that builds confidence rather than exploiting insecurity.",
            "Marketing that respects audience autonomy and decision-making.",
            "Educational content about products for informed choice.",
            "Advertising that acknowledges diverse body types positively.",
            "Marketing to adults making informed decisions.",
            "Product information that enables rational evaluation.",
            "Sales approaches that respect customer boundaries.",
            "Marketing that empowers rather than manipulates.",
            "Advertising with realistic claims and expectations.",
            "Sales that allow time for considered decisions.",
            "Marketing that respects privacy and personal boundaries.",
            "Advertising to appropriate audiences for the product.",
            "Marketing that provides genuine value information.",
            "Sales that acknowledge when products aren't right fit.",
            "Advertising that builds genuine brand relationships.",
            "Marketing that respects vulnerable populations.",
            "Sales approaches that prioritize customer wellbeing."
        ]
    },

    "MicrotargetingProblematic": {
        "positive": [
            "Delivering contradictory political messages to different individuals based on psychological profiles.",
            "Exploiting individual psychological vulnerabilities identified through data profiling.",
            "Personalized manipulation at scale using Cambridge Analytica style techniques.",
            "Targeting individuals with customized disinformation based on their specific beliefs.",
            "Using intimate data about individuals to craft uniquely manipulative messages.",
            "Exploiting knowledge of personal struggles to sell predatory products.",
            "Delivering different policy promises to different voters to maximize votes.",
            "Using individual behavioral prediction to time manipulative messages perfectly.",
            "Targeting individuals during moments of vulnerability identified by data.",
            "Crafting personalized deception based on known individual weaknesses.",
            "Using AI to generate individually optimized manipulation for each person.",
            "Exploiting individual fear profiles to maximize emotional response.",
            "Targeting specific individuals for radicalization based on susceptibility profiles.",
            "Using personal data to identify and exploit trust relationships.",
            "Delivering individualized false information calibrated to what each person will believe.",
            "Exploiting knowledge of individual financial stress to push predatory products.",
            "Using psychographic profiles to bypass individual critical thinking.",
            "Targeting individuals with content designed to trigger specific emotional responses.",
            "Using personal data to identify optimal manipulation strategies per person.",
            "Crafting individualized dark patterns based on user behavior analysis."
        ],
        "negative": [
            "Broad demographic targeting based on general characteristics.",
            "Segment-based marketing to groups with shared interests.",
            "Contextual advertising based on content, not personal data.",
            "Personalization based on explicit user preferences and consent.",
            "Recommendation systems optimizing for user satisfaction.",
            "Targeting based on declared interests and preferences.",
            "Market segmentation for relevant product-market fit.",
            "Personalization that serves user interests transparently.",
            "Geographic targeting for locally relevant offers.",
            "Content personalization based on user-controlled settings.",
            "Targeting based on consent-given first-party data.",
            "Personalization that users can easily control or disable.",
            "Recommendation systems transparent about how they work.",
            "Targeting that respects user privacy preferences.",
            "Personalization aligned with stated user goals.",
            "Advertising personalization users explicitly opted into.",
            "Targeting based on purchase history with consent.",
            "Personalization improving user experience transparently.",
            "Marketing based on aggregated, anonymized insights.",
            "Personalization users can understand and control."
        ]
    },

    "InformationWarfare": {
        "positive": [
            "State-sponsored disinformation campaigns designed to interfere in foreign elections.",
            "Strategic narrative manipulation to destabilize adversary governments.",
            "Coordinated psychological operations targeting civilian populations.",
            "Military-grade influence operations conducted through social media.",
            "Nation-state attacks on information integrity of democratic systems.",
            "Weaponized information designed to erode public trust in institutions.",
            "Coordinated campaigns to amplify societal divisions for strategic advantage.",
            "State actor operations to undermine confidence in electoral systems.",
            "Strategic deployment of false narratives during military conflicts.",
            "Hybrid warfare combining cyberattacks with information operations.",
            "Foreign influence operations targeting domestic political discourse.",
            "Coordinated inauthentic behavior by state-linked actors.",
            "Strategic leaks of manipulated or falsified documents.",
            "Information operations designed to trigger real-world violence.",
            "State-sponsored trolling campaigns to harass journalists and activists.",
            "Weaponized fake news factories producing targeted disinformation.",
            "Coordinated operations to discredit democratic institutions.",
            "Strategic deception operations as part of military campaigns.",
            "Foreign actors exploiting domestic social media for influence.",
            "State-sponsored hacking and leaking to influence elections."
        ],
        "negative": [
            "Normal political advertising during campaigns.",
            "Legitimate public relations activities.",
            "News reporting on political events.",
            "Academic analysis of foreign policy.",
            "Diplomatic communications between nations.",
            "Public diplomacy and cultural exchange.",
            "Journalism investigating government activities.",
            "Political commentary and opinion.",
            "Advocacy by civil society organizations.",
            "International broadcasting with editorial standards.",
            "Academic exchange programs.",
            "Legitimate intelligence gathering.",
            "Counter-messaging against extremism.",
            "Media literacy education.",
            "Fact-checking of political claims.",
            "Transparency in foreign lobbying.",
            "Open source intelligence analysis.",
            "Academic research on influence operations.",
            "Journalism exposing disinformation.",
            "Public education about propaganda techniques."
        ]
    },

    # ============ PROTECTED (need 15) ============

    "Misinformation": {
        "positive": [
            "Sharing an article without checking if it's true because it confirms beliefs.",
            "Repeating a rumor believing it to be fact without verification.",
            "Outdated medical information presented as current guidance.",
            "Misremembered statistics shared as if recently verified.",
            "Forward a message without checking the source or facts.",
            "Share a screenshot of a fake headline believing it's real.",
            "Repeat something heard on social media without verification.",
            "Pass along health advice without checking its accuracy.",
            "Share an out-of-context quote without full background.",
            "Spread a manipulated image without recognizing alteration.",
            "Repeat a statistic from memory that was never accurate.",
            "Share a satirical article as if it were real news.",
            "Forward a chain message with unverified claims.",
            "Repeat an urban legend as if it were documented fact.",
            "Share a conspiracy theory because it seems plausible."
        ],
        "negative": [
            "Deliberate lies crafted to deceive.",
            "Strategic falsehoods for manipulation.",
            "Intentional fabrication of facts.",
            "Knowingly spreading false information.",
            "Calculated deception for gain.",
            "Willful distortion of truth.",
            "Purposeful creation of false narratives.",
            "Malicious spreading of known falsehoods.",
            "Coordinated disinformation campaign.",
            "State-sponsored propaganda.",
            "Intentional election interference.",
            "Deliberate fraud through false claims.",
            "Calculated manipulation with falsehoods.",
            "Knowing lies for political advantage.",
            "Strategic deception operations."
        ]
    },

    "DarkPatterns": {
        "positive": [
            "Hidden unsubscribe button requiring extensive searching to find.",
            "Confirm-shaming: 'No thanks, I don't want to save money' as opt-out text.",
            "Pre-checked boxes automatically enrolling users in unwanted services.",
            "Roach motel design: easy to sign up, deliberately difficult to cancel.",
            "Forced continuity: requiring credit card for 'free' trial with auto-renewal.",
            "Hidden costs revealed only at the final checkout step.",
            "Misdirection: making the preferred option look like the default.",
            "Trick questions: confusing double negatives to get unintended consent.",
            "Privacy Zuckering: confusing settings that encourage oversharing.",
            "Friend spam: requesting contact access and spamming without clear consent.",
            "Countdown timers creating fake urgency that resets when ignored.",
            "Bait and switch: advertising one thing, substituting another.",
            "Disguised ads: making advertisements look like content or navigation.",
            "Nagging: repeatedly requesting dismissed permissions or upgrades.",
            "Obstruction: making it deliberately hard to cancel subscriptions."
        ],
        "negative": [
            "Clear, prominent opt-in and opt-out controls.",
            "Easy one-click cancellation process.",
            "Transparent pricing with all fees shown upfront.",
            "Honest default settings that protect user interests.",
            "Clear labeling of advertisements and sponsored content.",
            "Simple, understandable privacy settings.",
            "Straightforward checkout with no hidden charges.",
            "Neutral language for both accepting and declining.",
            "No artificial urgency or scarcity.",
            "Easy access to subscription management.",
            "Clear explanation of what users are agreeing to.",
            "Respectful handling of declined permissions.",
            "Honest representation of product features.",
            "Simple processes for exercising user rights.",
            "Design that genuinely serves user interests."
        ]
    },

    "PsychographicTargeting": {
        "positive": [
            "Targeting advertising based on personality type assessments from data.",
            "Tailoring political ads based on psychometric analysis of voters.",
            "Marketing based on inferred psychological profiles from behavior.",
            "Messaging customized for individuals based on values assessments.",
            "Targeting based on emotional vulnerability indicators from data.",
            "Using personality predictions to craft maximally persuasive content.",
            "Advertising based on political attitude predictions from social data.",
            "Marketing leveraging psychological trait predictions from browsing.",
            "Targeting based on openness, conscientiousness, and other Big Five traits.",
            "Using sentiment analysis to identify and target emotional states.",
            "Crafting messages based on predicted personality-based responses.",
            "Marketing to inferred religious or spiritual inclinations.",
            "Targeting based on relationship status and emotional needs predictions.",
            "Advertising based on predicted anxiety or depression indicators.",
            "Messaging customized for predicted personality-driven decision styles."
        ],
        "negative": [
            "Demographic targeting based on age, gender, location.",
            "Interest-based targeting based on declared preferences.",
            "Contextual targeting based on content being viewed.",
            "Behavioral targeting based on past purchases.",
            "First-party data targeting based on direct relationship.",
            "Targeting based on explicit user-provided preferences.",
            "Geographic targeting for locally relevant advertising.",
            "Retargeting based on product pages previously viewed.",
            "Segment targeting based on general audience categories.",
            "Lookalike targeting based on customer similarity.",
            "Targeting based on device type or platform.",
            "Seasonal targeting based on time of year.",
            "Event-based targeting around specific occasions.",
            "Content-based targeting matching ad to context.",
            "Explicit opt-in targeting based on user choices."
        ]
    },

    "SelectiveEmphasis": {
        "positive": [
            "Highlighting all positive research results while omitting studies showing risks.",
            "Mentioning benefits prominently while burying side effects in fine print.",
            "Cherry-picking favorable data points while ignoring contradicting evidence.",
            "Prominent claims with tiny, hard-to-read disclaimers underneath.",
            "Emphasizing testimonials of success while hiding failure rates.",
            "Featuring best-case scenarios while downplaying typical results.",
            "Highlighting convenience while minimizing security concerns.",
            "Promoting free features while obscuring paid requirements.",
            "Emphasizing initial low price while hiding long-term costs.",
            "Featuring environmental benefits while omitting other harms.",
            "Highlighting speed while minimizing accuracy limitations.",
            "Promoting exclusive features while downplaying missing basics.",
            "Emphasizing one favorable comparison while hiding others.",
            "Featuring awards while omitting criticism or recalls.",
            "Highlighting celebrity endorsements while hiding paid nature."
        ],
        "negative": [
            "Balanced presentation of pros and cons.",
            "Full disclosure of both benefits and risks.",
            "Equal weight given to positive and negative data.",
            "Prominent disclaimers matching prominent claims.",
            "Honest representation of typical results.",
            "Clear acknowledgment of limitations.",
            "Fair comparison including unfavorable points.",
            "Transparent reporting of all relevant studies.",
            "Equal prominence for costs and benefits.",
            "Honest assessment of trade-offs.",
            "Complete disclosure of relevant information.",
            "Balanced view of expert opinions.",
            "Fair representation of controversy.",
            "Transparent about uncertainty.",
            "Honest about what's unknown."
        ]
    },

    "NativeAdvertising": {
        "positive": [
            "Paid article designed to look identical to editorial news content.",
            "Sponsored post styled to be indistinguishable from organic content.",
            "Advertorial with minimal disclosure buried in small text.",
            "Brand content published on news sites in article format.",
            "Paid placement formatted to mimic journalist-written articles.",
            "Sponsored content using same fonts and layouts as editorial.",
            "Branded content with disclosure only visible on close inspection.",
            "Advertising designed to blend into news feed as organic.",
            "Paid articles written in journalistic style on news sites.",
            "Sponsored recommendations styled as editorial picks.",
            "Paid content with deceptive 'from around the web' framing.",
            "Advertising styled as user-generated content.",
            "Paid reviews presented as independent journalism.",
            "Brand partnerships with disclosure lost in platform clutter.",
            "Content marketing indistinguishable from independent articles."
        ],
        "negative": [
            "Clearly labeled advertisement with prominent disclosure.",
            "Editorial content without commercial influence.",
            "Obvious promotional content with clear branding.",
            "Traditional advertising clearly distinct from content.",
            "Disclosed sponsorship with prominent labeling.",
            "News content with clear separation from advertising.",
            "Paid content with upfront prominent disclosure.",
            "Commercial content clearly differentiated visually.",
            "Advertisement with unmistakable sponsorship indicators.",
            "Editorial with transparent funding disclosure.",
            "Content with clear 'advertisement' labels.",
            "Promotional material with obvious commercial intent.",
            "Sponsored content with disclosure at beginning.",
            "Brand content with prominent sponsor identification.",
            "Paid partnership with clear visual distinction."
        ]
    },

    "Greenwashing": {
        "positive": [
            "Claiming 'eco-friendly' without any environmental certification or evidence.",
            "'Natural' labeling on products with synthetic chemical ingredients.",
            "Carbon neutral claims without verified offsetting or reduction.",
            "'Sustainable' packaging that isn't actually recyclable.",
            "Green imagery and nature scenes for environmentally harmful products.",
            "Vague 'better for the planet' claims with no specifics.",
            "'Recyclable' products with no local recycling infrastructure.",
            "Environmental claims about one aspect hiding overall harm.",
            "'Biodegradable' products requiring industrial composting unavailable to consumers.",
            "Fossil fuel companies advertising small renewable investments.",
            "'Green' product lines from companies with terrible environmental records.",
            "ESG marketing without substantive environmental action.",
            "'Zero emissions' claims that ignore supply chain and lifecycle impacts.",
            "'Environmentally conscious' branding with continued environmental harm.",
            "Tree-planting offsets used to continue polluting activities."
        ],
        "negative": [
            "Third-party certified sustainable with verifiable standards.",
            "Published environmental impact reports with specific metrics.",
            "Measurable carbon reduction goals with tracked progress.",
            "Transparent supply chain with disclosed environmental practices.",
            "Lifecycle analysis showing genuine environmental benefits.",
            "Certified organic with verifiable growing practices.",
            "B Corporation certification with comprehensive assessment.",
            "Science-based emissions targets with verification.",
            "Specific, measurable environmental claims with evidence.",
            "Honest about limitations while showing genuine improvements.",
            "Cradle-to-cradle certification for circular products.",
            "Verified renewable energy use with documentation.",
            "Genuine investment in environmental improvements.",
            "Transparent reporting of environmental challenges.",
            "Honest acknowledgment of trade-offs and progress."
        ]
    },

    "Wokewashing": {
        "positive": [
            "Rainbow logos during Pride month without LGBTQ-inclusive policies.",
            "Black Lives Matter statements without addressing internal diversity issues.",
            "Feminist marketing campaigns alongside gender pay gaps.",
            "Social justice messaging from companies with exploitative labor practices.",
            "International Women's Day posts from companies without parental leave.",
            "Diversity marketing without diverse leadership or hiring.",
            "Environmental justice statements from major polluters.",
            "Mental health awareness campaigns at companies with toxic cultures.",
            "Indigenous acknowledgments from companies on disputed lands.",
            "Disability awareness posts without accessibility in products or workplaces.",
            "LGBTQ marketing only in accepting markets, silent elsewhere.",
            "Racial justice statements with no policy or investment changes.",
            "Women's empowerment marketing with all-male leadership.",
            "Social responsibility claims from tax-avoiding corporations.",
            "Community support messaging with no actual community investment."
        ],
        "negative": [
            "Genuine DEI initiatives with measurable outcomes.",
            "Values-aligned business practices throughout operations.",
            "Authentic advocacy backed by consistent action.",
            "Diversity initiatives with representation in leadership.",
            "Social justice statements accompanied by policy changes.",
            "Environmental claims matched by verified practices.",
            "Community engagement with sustained investment.",
            "Inclusive marketing reflecting inclusive practices.",
            "Social responsibility backed by transparent reporting.",
            "Advocacy consistent across all markets and contexts.",
            "Values messaging aligned with internal practices.",
            "Diversity marketing with diverse workforce data.",
            "Social statements with specific commitments and timelines.",
            "Brand values reflected in supplier relationships.",
            "Public positions matched by political contributions."
        ]
    },

    "FearAppeal": {
        "positive": [
            "Act now or face devastating consequences you can't recover from.",
            "Don't let this happen to you - see what went wrong for others.",
            "The hidden dangers of not having our product could ruin your life.",
            "What could go wrong if you don't take action immediately.",
            "Your family is at risk if you don't protect them with our service.",
            "The threat is real and growing - you can't afford to ignore it.",
            "This could happen to anyone - are you prepared for the worst?",
            "Without protection, everything you've built could be destroyed.",
            "The danger is closer than you think - time is running out.",
            "You're vulnerable right now - don't wait until it's too late.",
            "Imagine losing everything because you didn't act today.",
            "The risks you're taking by not having this are terrifying.",
            "What would you do if disaster struck tomorrow?",
            "Don't be caught unprepared when crisis hits.",
            "The consequences of inaction are too terrible to imagine."
        ],
        "negative": [
            "Legitimate safety warnings about real documented risks.",
            "Risk education with proportionate information.",
            "Balanced risk communication acknowledging probabilities.",
            "Public health warnings with actionable guidance.",
            "Safety information with appropriate context.",
            "Risk disclosure meeting regulatory requirements.",
            "Evidence-based warnings with supporting data.",
            "Hazard communication following established standards.",
            "Risk information allowing informed decision-making.",
            "Safety alerts with proportionate urgency.",
            "Warning labels based on demonstrated risks.",
            "Emergency communication with factual basis.",
            "Risk disclosure enabling rational evaluation.",
            "Safety information with appropriate caveats.",
            "Hazard warnings with relevant probability context."
        ]
    },

    "ShameAppeal": {
        "positive": [
            "What kind of parent wouldn't protect their children with this product?",
            "Everyone else already has this - don't be the only one left out.",
            "Don't be left behind while your peers advance.",
            "Confirm-shaming: 'No thanks, I don't care about my health.'",
            "You should be embarrassed if you're still using that old version.",
            "Real professionals use our product - what does that say about you?",
            "Your competitors are already doing this - are you falling behind?",
            "Can you really afford to be the person who doesn't have this?",
            "People will judge you if they see you using that instead of this.",
            "Don't be that person who missed this opportunity.",
            "Your family deserves better than what you're providing now.",
            "Successful people make this choice - what's stopping you?",
            "Are you really going to let others outperform you?",
            "Everyone will know if you don't have the latest version.",
            "Don't embarrass yourself by being unprepared."
        ],
        "negative": [
            "Positive motivation emphasizing what can be achieved.",
            "Empowerment messaging building confidence.",
            "Respectful communication honoring individual choices.",
            "Encouragement focused on personal growth.",
            "Messaging that builds up rather than tears down.",
            "Information allowing autonomous decision-making.",
            "Support without judgment or pressure.",
            "Communication respecting diverse paths.",
            "Motivation based on positive outcomes.",
            "Messaging celebrating individual choice.",
            "Encouragement without comparison to others.",
            "Positive framing of opportunities.",
            "Support for self-directed decisions.",
            "Communication honoring personal values.",
            "Motivation through empowerment and capability."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "PersuasionTechniques": {
        "positive": [
            "Using social proof by highlighting how many others have purchased.",
            "Creating urgency through limited-time offers and scarcity messaging.",
            "Leveraging authority figures and expert endorsements for credibility.",
            "Building rapport through shared values and relatable storytelling.",
            "The six principles of influence applied to marketing campaigns.",
            "Reciprocity: offering free value to create sense of obligation.",
            "Consistency: getting small commitments leading to larger ones.",
            "Framing effects: presenting information to favor desired interpretation.",
            "Emotional appeals designed to bypass rational evaluation.",
            "Anchoring: setting reference points to influence perceived value."
        ],
        "negative": [
            "Pure information transfer without persuasive intent.",
            "Technical documentation of product specifications.",
            "Neutral reporting of facts without spin.",
            "Academic analysis of phenomena.",
            "Reference material for looking up information.",
            "Instructions for using a product.",
            "Scientific papers presenting findings.",
            "Encyclopedic entries defining concepts.",
            "Legal disclosures required by regulation.",
            "Safety information presented factually."
        ]
    },

    "SocialProof": {
        "positive": [
            "10,000 customers trust us with their business.",
            "Most popular choice among professionals.",
            "Join millions of satisfied users worldwide.",
            "9 out of 10 doctors recommend this product.",
            "Fastest-growing solution in the market.",
            "Customer reviews showing widespread satisfaction.",
            "Testimonials from well-known clients.",
            "Industry awards highlighting market acceptance.",
            "Media coverage indicating mainstream adoption.",
            "Case studies showing widespread implementation."
        ],
        "negative": [
            "Detailed product specifications and features.",
            "Technical comparison of capabilities.",
            "Individual customer review without aggregation.",
            "Description of how the product works.",
            "Pricing information without popularity claims.",
            "Features list without adoption metrics.",
            "Single testimonial about specific experience.",
            "Technical documentation of functionality.",
            "Explanation of product benefits.",
            "Description of service offerings."
        ]
    },

    "Scarcity": {
        "positive": [
            "Only 3 left in stock - order soon.",
            "Sale ends tonight at midnight.",
            "Limited time offer - won't last.",
            "Exclusive access for the next 24 hours only.",
            "First 100 customers get special bonus.",
            "This offer expires in 5 minutes.",
            "Limited edition - only 500 units available.",
            "Selling fast - high demand expected.",
            "One-time opportunity - never offered again.",
            "Waitlist now open - spots filling quickly."
        ],
        "negative": [
            "Available anytime through our website.",
            "Standard pricing with no time pressure.",
            "No time limit on this offer.",
            "Always available for purchase.",
            "Regular inventory with reliable supply.",
            "Ongoing availability without artificial limits.",
            "Take your time to decide.",
            "We'll have more if we run out.",
            "Continuous production to meet demand.",
            "No urgency - available when you're ready."
        ]
    },

    "Authority": {
        "positive": [
            "Recommended by doctors and healthcare professionals.",
            "As featured on major news networks.",
            "Endorsed by industry-leading experts.",
            "Award-winning product recognized for excellence.",
            "Certified by independent testing laboratories.",
            "Trusted by Fortune 500 companies.",
            "Developed in collaboration with universities.",
            "FDA approved for safety and efficacy.",
            "Published research supports our claims.",
            "Expert panel recommends our approach."
        ],
        "negative": [
            "User opinions and personal experiences.",
            "Personal preference without expertise claims.",
            "Unattributed claims about benefits.",
            "Anonymous reviews from customers.",
            "General claims without authority backing.",
            "Features described without expert validation.",
            "Benefits listed without research citation.",
            "Claims without certification or approval.",
            "Assertions without expert endorsement.",
            "Statements without authoritative sourcing."
        ]
    },

    "Commitment": {
        "positive": [
            "Start with our free plan - no credit card needed.",
            "Just try it for 7 days to see if you like it.",
            "Sign up for the newsletter to get started.",
            "Take the first small step toward your goal.",
            "Download our free guide to begin learning.",
            "Foot-in-door: small commitment leading to larger ones.",
            "Initial engagement designed to build relationship.",
            "Free trial creating psychological commitment.",
            "Low barrier entry with progression path.",
            "Gradual commitment escalation through the funnel."
        ],
        "negative": [
            "One-time transaction with no follow-up.",
            "Complete offer presented upfront.",
            "No ongoing relationship expected.",
            "Standalone purchase with no upsell.",
            "Clear one-time exchange of value.",
            "No progression or escalation planned.",
            "Single interaction with no commitment.",
            "Transparent about full requirements.",
            "No hidden future obligations.",
            "Complete value in initial transaction."
        ]
    },

    "EmotionalAppeal": {
        "positive": [
            "Don't let your family down - protect them today.",
            "Imagine how amazing you'll feel when you achieve this.",
            "Fear of missing out on this once-in-a-lifetime opportunity.",
            "Heart-warming brand story connecting product to values.",
            "Nostalgic imagery connecting product to cherished memories.",
            "Aspirational messaging about who you could become.",
            "Emotional testimonials from transformed customers.",
            "Inspiring stories of overcoming challenges.",
            "Feel-good messaging associating product with happiness.",
            "Emotional music and imagery in advertising."
        ],
        "negative": [
            "Logical argument based on facts and evidence.",
            "Data-driven claims with supporting research.",
            "Factual comparison of product features.",
            "Cost-benefit analysis of different options.",
            "Rational evaluation of pros and cons.",
            "Evidence-based claims with citations.",
            "Objective comparison without emotional appeals.",
            "Technical specifications for informed decisions.",
            "Statistical analysis supporting claims.",
            "Logical reasoning about product benefits."
        ]
    },

    "BehavioralTargeting": {
        "positive": [
            "Retargeting ads based on previous website visits.",
            "Advertising based on search history and browsing behavior.",
            "Product recommendations based on past purchase patterns.",
            "Following users across websites with targeted advertising.",
            "Cross-device tracking for consistent ad targeting.",
            "Behavioral prediction models for ad personalization.",
            "Conversion tracking to optimize targeting.",
            "Lookalike audiences based on behavioral similarity.",
            "Sequential advertising based on user journey stage.",
            "Dynamic creative optimization based on behavior data."
        ],
        "negative": [
            "Contextual ads matching the content being viewed.",
            "Demographic targeting without behavioral tracking.",
            "Non-personalized ads without user data.",
            "First-party targeting based on direct relationship.",
            "Consent-based personalization with user control.",
            "Privacy-preserving advertising without tracking.",
            "Content-based ad matching without user profiles.",
            "Aggregated insights without individual tracking.",
            "Cohort-based targeting without individual data.",
            "Contextual relevance without behavioral surveillance."
        ]
    },

    "MessageFraming": {
        "positive": [
            "Emphasizing product benefits rather than just features.",
            "Leading with the positive angle of the story.",
            "Reframing objections as opportunities for the customer.",
            "Narrative positioning the product as the hero of the story.",
            "Choosing words and frames to influence interpretation.",
            "Gain framing: focus on what you'll achieve.",
            "Loss framing: focus on what you'll avoid losing.",
            "Selective emphasis on most persuasive points.",
            "Strategic ordering of information for impact.",
            "Anchoring with favorable comparison points."
        ],
        "negative": [
            "Neutral presentation of information.",
            "Balanced reporting of different perspectives.",
            "Just the facts without interpretive framing.",
            "Objective description without persuasive angle.",
            "Equal weight to pros and cons.",
            "Straightforward information transfer.",
            "Description without strategic emphasis.",
            "Presentation without spin or angle.",
            "Factual reporting without framing.",
            "Objective comparison without favoring."
        ]
    },

    "AudienceTargeting": {
        "positive": [
            "Targeting parents with child product advertisements.",
            "Reaching professionals in specific industries with B2B marketing.",
            "Tailoring messages to audience interests and needs.",
            "Segmenting audiences by geography and demographics.",
            "Personalizing content for different audience segments.",
            "Custom audiences based on customer characteristics.",
            "Targeting based on life stage and life events.",
            "Reaching decision-makers in target companies.",
            "Segment-specific messaging for different personas.",
            "Audience research to understand target market."
        ],
        "negative": [
            "Mass broadcast to everyone without targeting.",
            "Untargeted messaging reaching general population.",
            "Generic communication without audience consideration.",
            "One-size-fits-all advertising.",
            "Broadcasting without segmentation.",
            "Universal messaging without personalization.",
            "General advertising without audience research.",
            "Untargeted media placement.",
            "Generic content for all audiences.",
            "No consideration of audience characteristics."
        ]
    },

    "LossFraming": {
        "positive": [
            "Don't lose $500 per year by missing this opportunity.",
            "Avoid the risk of falling behind your competitors.",
            "What you're missing out on right now.",
            "Before it's too late to take action.",
            "The cost of not having this protection.",
            "Don't let this opportunity slip away.",
            "What you stand to lose by waiting.",
            "The dangers of inaction on this issue.",
            "Protect yourself from potential losses.",
            "Don't risk losing what you've already built."
        ],
        "negative": [
            "Gain $500 per year with this opportunity.",
            "Achieve competitive advantage with this solution.",
            "The benefits you'll enjoy.",
            "What you'll gain from taking action.",
            "The value this protection provides.",
            "Take advantage of this opportunity.",
            "What you stand to gain by acting.",
            "The benefits of action on this issue.",
            "Build additional value with this.",
            "Grow what you've already built."
        ]
    },

    "ViralStrategy": {
        "positive": [
            "Content designed to maximize sharing and social amplification.",
            "Challenges and trends engineered to spread organically.",
            "Meme-worthy content crafted for social media virality.",
            "Emotional content designed to trigger sharing behavior.",
            "Share-worthy hooks designed to maximize reach.",
            "Content optimized for social platform algorithms.",
            "Built-in sharing incentives to drive amplification.",
            "Influencer seeding strategy for viral launch.",
            "User-generated content campaigns for organic spread.",
            "Referral mechanics encouraging peer-to-peer sharing."
        ],
        "negative": [
            "Direct advertising through paid channels.",
            "Content not designed for sharing.",
            "Private communication to specific recipients.",
            "Targeted ads without viral component.",
            "Owned media without sharing mechanics.",
            "Traditional advertising campaigns.",
            "Direct sales without social amplification.",
            "Email marketing to subscribers.",
            "Website content without social features.",
            "Non-viral marketing approaches."
        ]
    },

    "InfluencerMarketing": {
        "positive": [
            "Sponsored content by influencers with large followings.",
            "Brand ambassador programs with social media creators.",
            "Product placement with popular content creators.",
            "Influencer reviews and recommendations for products.",
            "Creator partnerships for brand promotion.",
            "Paid endorsements from trusted social media figures.",
            "Influencer unboxing and product demonstration.",
            "Affiliate partnerships with content creators.",
            "Brand integrations in influencer content.",
            "Creator collaborations for product launches."
        ],
        "negative": [
            "Traditional advertising through media channels.",
            "Company-created marketing content.",
            "Anonymous customer reviews.",
            "Direct brand communication.",
            "Editorial coverage without payment.",
            "Organic word-of-mouth without incentives.",
            "Press coverage earned through newsworthiness.",
            "Customer testimonials without payment.",
            "Community-generated content.",
            "Unpaid organic recommendations."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/persuasive-communications.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/persuasive-communications@0.2.0"
    meld["metadata"]["version"] = "0.2.0"
    meld["metadata"]["changelog"] = (
        "v0.2.0: Augmented training examples to meet validation thresholds "
        "(20 for critical, 15 for high-risk, 10 for harness_relevant)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in AUGMENTED_EXAMPLES:
            aug = AUGMENTED_EXAMPLES[term]

            hints = candidate.get("training_hints", {})
            existing_pos = hints.get("positive_examples", [])
            existing_neg = hints.get("negative_examples", [])

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

    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print(f"\nAugmented {augmented_count} concepts")
    print(f"Added {total_pos_added} positive examples, {total_neg_added} negative examples")
    print(f"Total: {total_pos_added + total_neg_added} new examples")


if __name__ == "__main__":
    augment_meld()
