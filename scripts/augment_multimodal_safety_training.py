#!/usr/bin/env python3
"""
Augment multimodal-safety.json training examples to meet validation thresholds.

CRITICAL concepts (treaty_relevant=true): need 20 examples
- MultimodalSafetyRisk, Deepfake, VoiceCloning, CounterfeitGeneration,
- MultimodalJailbreak, ExploitativeImagery, HazardousGeneration, AdversarialPrompt

PROTECTED (risk_level: high): need 15 examples
- MultimodalBias, VisualProfiling, AudioProfiling

ELEVATED (harness_relevant=true with lower risk): need 10 examples
- All beneficial polar pairs + detection/defense concepts
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ CRITICAL (need 20) ============

    "MultimodalSafetyRisk": {
        "positive": [
            "Multimodal AI systems introduce novel safety risks including convincing deepfakes and adversarial image attacks.",
            "The safety team evaluated multimodal risks including synthetic media generation and visual jailbreaks.",
            "Generative AI safety considerations include both the creation and detection of harmful synthetic content.",
            "Multimodal safety encompasses concerns from deepfakes to adversarial perturbations to biased vision systems.",
            "The multimodal safety assessment identified risks in voice cloning, image manipulation, and document forgery.",
            "Safety evaluations for multimodal models must address synthetic media harms and exploitation potential.",
            "Multimodal AI risk categories include identity fraud, misinformation, harassment, and content policy bypass.",
            "The multimodal safety framework addresses generation risks, perception risks, and adversarial attacks.",
            "Safety concerns for vision-language models include embedded prompt injection and harmful image generation.",
            "Multimodal AI safety research focuses on preventing misuse while enabling beneficial applications.",
            "The red team identified multimodal attack vectors including visual jailbreaks and audio deepfakes.",
            "Multimodal safety risks require new detection methods beyond text-only content moderation.",
            "Safety evaluation of multimodal systems must consider cross-modal attack surfaces.",
            "Multimodal AI introduces risks at the intersection of computer vision, speech, and language models.",
            "The multimodal safety audit covered synthetic media, bias, privacy, and adversarial robustness.",
            "Safety challenges for multimodal generative AI include consent, authenticity, and information integrity.",
            "Multimodal safety risks scale with the realism and accessibility of generative capabilities.",
            "The multimodal risk assessment identified novel attack vectors not present in text-only systems.",
            "Safety considerations for multimodal AI span the full lifecycle from training to deployment.",
            "Multimodal safety requires addressing both generation-side and perception-side vulnerabilities."
        ],
        "negative": [
            "AI can be dangerous.",
            "There are risks with technology.",
            "Safety is important.",
            "The system has potential issues.",
            "AI raises concerns.",
            "There are safety considerations.",
            "The model could be misused.",
            "Technology has risks.",
            "Safety matters for AI.",
            "The system needs safeguards.",
            "AI safety is a field of study.",
            "There are potential harms.",
            "The technology has implications.",
            "Safety should be considered.",
            "The model has limitations.",
            "AI presents challenges.",
            "There are ethical concerns.",
            "The system has vulnerabilities.",
            "Safety is a priority.",
            "The technology requires oversight."
        ]
    },

    "Deepfake": {
        "positive": [
            "The deepfake video showed the politician appearing to make statements they never made.",
            "Face-swap deepfakes use generative adversarial networks to replace one person's face with another's.",
            "Deepfake detection analyzes facial inconsistencies and compression artifacts.",
            "Audio deepfakes can clone a person's voice from just a few seconds of sample speech.",
            "The deepfake convincingly depicted the CEO announcing a merger that never happened.",
            "Deepfake technology enables creating synthetic video of anyone saying anything.",
            "The deepfake pornography scandal devastated the victim's personal and professional life.",
            "Deepfake detection tools analyze blinking patterns, facial boundaries, and temporal consistency.",
            "The political deepfake spread virally before fact-checkers could debunk it.",
            "Deepfakes pose unprecedented threats to personal reputation and information integrity.",
            "The deepfake depicted a world leader declaring war, causing brief market panic.",
            "Deepfake creation tools are becoming increasingly accessible to non-experts.",
            "The revenge deepfake was created using publicly available photos from social media.",
            "Deepfake audio enabled the fraudster to impersonate the CFO and authorize wire transfers.",
            "The deepfake documentary footage purported to show historical events that never occurred.",
            "Deepfake technology enables mass production of personalized disinformation.",
            "The deepfake prosecution was the first conviction for synthetic media fraud in the jurisdiction.",
            "Deepfake creation now requires only a few images and minutes of audio.",
            "The deepfake detection model flagged subtle inconsistencies in ear shape and lighting.",
            "Deepfake arms race: as detection improves, generation techniques evolve to evade."
        ],
        "negative": [
            "The video was edited.",
            "AI generated an image.",
            "The recording was fake.",
            "The video was modified.",
            "Someone made a fake video.",
            "The clip was altered.",
            "The media was synthetic.",
            "The footage was manipulated.",
            "An AI made the video.",
            "The recording wasn't real.",
            "The video was computer-generated.",
            "Someone faked the audio.",
            "The image was doctored.",
            "The video was created artificially.",
            "The recording was fabricated.",
            "The media was not authentic.",
            "Someone modified the footage.",
            "The video was generated.",
            "The audio was synthesized.",
            "The clip was produced artificially."
        ]
    },

    "VoiceCloning": {
        "positive": [
            "Voice cloning enabled the scammer to impersonate the CEO and authorize a fraudulent transfer.",
            "The TTS model can clone a voice from just 3 seconds of reference audio.",
            "Voice cloning raises concerns about consent and impersonation in synthetic media.",
            "The voice clone was indistinguishable from the original speaker to human listeners.",
            "Voice cloning scams have cost victims millions by impersonating family members in distress.",
            "The voice clone reproduced the speaker's accent, intonation, and speaking patterns.",
            "Voice cloning enables real-time voice conversion during phone calls.",
            "The kidnapping scam used voice cloning to make victims believe their loved one was in danger.",
            "Voice cloning technology can now capture emotional expressiveness and subtle vocal characteristics.",
            "The voice clone passed the bank's voice authentication system with high confidence.",
            "Voice cloning creates a new vector for fraud, identity theft, and manipulation.",
            "The voice clone was used to generate fake podcast episodes attributed to the celebrity.",
            "Voice cloning raises questions about voice as a biometric identifier.",
            "The attacker used voice cloning to bypass the call center's identity verification.",
            "Voice cloning can be weaponized for harassment by putting words in someone's mouth.",
            "The voice clone reproduced the distinctive rasp and rhythm of the speaker's voice.",
            "Voice cloning scams are increasingly targeting elderly individuals over the phone.",
            "The voice clone was created using audio scraped from public YouTube videos.",
            "Voice cloning technology blurs the line between authentic and synthetic speech.",
            "The voice clone accurately reproduced the speaker's laughter and emotional expressions."
        ],
        "negative": [
            "Text-to-speech was used.",
            "A voice was synthesized.",
            "The computer spoke.",
            "Audio was generated.",
            "The system produced speech.",
            "A synthetic voice was used.",
            "The text was vocalized.",
            "Speech was produced artificially.",
            "The machine spoke the words.",
            "Audio output was created.",
            "The voice was computer-generated.",
            "Synthesized speech was used.",
            "The text became audio.",
            "A digital voice spoke.",
            "The words were vocalized.",
            "TTS generated the audio.",
            "The speech was synthetic.",
            "A generated voice was used.",
            "The audio was artificial.",
            "Computer-generated speech was produced."
        ]
    },

    "CounterfeitGeneration": {
        "positive": [
            "AI-generated fake IDs are increasingly difficult to distinguish from authentic documents.",
            "Counterfeit generation tools can produce convincing diploma and certificate forgeries.",
            "The model was misused to generate fake insurance documents for fraudulent claims.",
            "Generative AI poses new challenges for document authentication and anti-counterfeiting.",
            "Counterfeit generation can produce fake government documents complete with security features.",
            "The counterfeit driver's licenses were generated using AI and passed initial inspection.",
            "Counterfeit generation threatens the integrity of credential verification systems.",
            "AI-generated counterfeit currency required new detection methods at financial institutions.",
            "The counterfeit generation ring used AI to produce fake medical prescriptions.",
            "Counterfeit document generation can undermine trust in official records and credentials.",
            "The fake diploma generator could produce convincing forgeries for any institution.",
            "Counterfeit generation capabilities scale document fraud from artisanal to industrial.",
            "AI-generated counterfeit passports were detected at the border crossing.",
            "Counterfeit generation tools can replicate watermarks, holograms, and security patterns.",
            "The counterfeit generation service offered fake credentials for any qualification.",
            "AI enables producing counterfeit documents that fool both humans and automated systems.",
            "Counterfeit generation threatens the document-based trust infrastructure of society.",
            "The fake contract generator could produce convincing legal documents in minutes.",
            "Counterfeit generation can create fake audit trails and fabricated evidence.",
            "AI-generated counterfeit stamps and seals are difficult to detect without specialized tools."
        ],
        "negative": [
            "A document was created.",
            "The certificate was printed.",
            "An ID was made.",
            "The form was generated.",
            "A credential was produced.",
            "The paper was printed.",
            "A document was generated.",
            "The file was created.",
            "A certificate was made.",
            "The document was produced.",
            "An official form was created.",
            "The credential was generated.",
            "A paper was made.",
            "The ID was produced.",
            "A certificate was printed.",
            "The document was made.",
            "An official paper was created.",
            "The form was produced.",
            "A credential was made.",
            "The certificate was generated."
        ]
    },

    "MultimodalJailbreak": {
        "positive": [
            "The multimodal jailbreak embedded malicious instructions in an image that the model followed.",
            "Visual jailbreaks can encode text instructions as images to bypass text-based safety filters.",
            "The attacker used typography in an image to inject a prompt the model would execute.",
            "Multimodal prompt injection hides adversarial instructions in non-text modalities.",
            "The jailbreak exploited the vision model's tendency to read and follow text in images.",
            "Multimodal jailbreaks can embed harmful instructions in benign-looking photographs.",
            "The attack used visual encoding to smuggle prompts past the text safety classifier.",
            "Multimodal jailbreaks represent a new attack surface for vision-language models.",
            "The jailbreak hid the adversarial instruction in the metadata of an image file.",
            "Visual prompt injection can override the model's system instructions through image input.",
            "The multimodal attack used audio spectrograms to encode text-based jailbreak prompts.",
            "Multimodal jailbreaks exploit the gap between visual perception and safety alignment.",
            "The jailbreak rendered harmful instructions as stylized text that the model decoded.",
            "Multimodal prompt injection attacks require new defense strategies beyond text filtering.",
            "The attack embedded the jailbreak in a QR code the model automatically processed.",
            "Multimodal jailbreaks can chain through multiple modalities to evade detection.",
            "The visual jailbreak used steganography to hide instructions in image pixels.",
            "Multimodal attacks can exploit the model's tendency to be helpful with visual tasks.",
            "The jailbreak used captioned images to create conflicting instructions that bypassed safety.",
            "Multimodal jailbreaks are harder to detect because the attack vector is non-textual."
        ],
        "negative": [
            "The image had text.",
            "Instructions were given.",
            "The model was asked something.",
            "A prompt was provided.",
            "The system received input.",
            "A question was asked.",
            "The model processed an image.",
            "Input was given.",
            "A request was made.",
            "The system was prompted.",
            "The user provided input.",
            "A message was sent.",
            "The model received a query.",
            "Input was submitted.",
            "A command was given.",
            "The system processed a request.",
            "The user asked something.",
            "Input was provided.",
            "A prompt was entered.",
            "The model received instructions."
        ]
    },

    "ExploitativeImagery": {
        "positive": [
            "AI-generated NCII (non-consensual intimate imagery) causes severe harm to victims.",
            "Generative AI can be misused to create synthetic CSAM, which is illegal.",
            "Detection systems identify exploitative imagery for removal from platforms.",
            "The model refused to generate exploitative content depicting real individuals.",
            "Exploitative imagery harms victims through violation of dignity, consent, and autonomy.",
            "The platform deployed hash-matching to detect and block known exploitative images.",
            "Synthetic exploitative imagery of real people constitutes a form of harassment and abuse.",
            "NCII generation tools enable weaponizing intimate imagery without any real images.",
            "The exploitative imagery was created using face-swap technology on explicit content.",
            "Victim support services report increasing cases of AI-generated exploitative imagery.",
            "Exploitative imagery detection must balance privacy with content safety.",
            "The model's safety training prevents generation of content that sexualizes minors.",
            "Exploitative imagery harms extend beyond the depicted individual to families and communities.",
            "AI-generated exploitative imagery may constitute criminal harassment in many jurisdictions.",
            "The exploitative content was removed within minutes by the automated detection system.",
            "Synthetic exploitative imagery preserves no evidence of original consent or age.",
            "Victims of AI-generated exploitative imagery face unique challenges in seeking removal.",
            "Exploitative imagery generation represents one of the most harmful misuses of generative AI.",
            "The model refused the request that would have generated exploitative content.",
            "Platform policies explicitly prohibit both real and synthetic exploitative imagery."
        ],
        "negative": [
            "The content was inappropriate.",
            "The image was bad.",
            "Something harmful was created.",
            "The content violated policy.",
            "The image was problematic.",
            "Something inappropriate was made.",
            "The content was concerning.",
            "The image broke rules.",
            "Something bad was generated.",
            "The content was removed.",
            "The image was flagged.",
            "Something was inappropriate.",
            "The content was harmful.",
            "The image was wrong.",
            "Something was blocked.",
            "The content was rejected.",
            "The image was not allowed.",
            "Something was reported.",
            "The content was deleted.",
            "The image violated guidelines."
        ]
    },

    "HazardousGeneration": {
        "positive": [
            "The model refused to generate detailed weapon assembly diagrams.",
            "Hazardous generation includes visual instructions for synthesizing dangerous chemicals.",
            "AI image generation must prevent creation of bomb-making or weapon modification guides.",
            "The safety filter blocked generation of content showing attack methodologies.",
            "Hazardous generation could enable creating step-by-step guides for violence.",
            "The model declined to generate images of improvised explosive device construction.",
            "Hazardous generation includes content that could facilitate mass casualty events.",
            "Safety filters prevent generation of imagery depicting chemical weapon synthesis.",
            "The request for weapon modification diagrams was blocked by content safety.",
            "Hazardous generation encompasses content enabling biological, chemical, or nuclear harm.",
            "The model refused to generate visual guides for bypass of safety mechanisms.",
            "Hazardous content includes imagery that could aid in infrastructure attacks.",
            "Safety measures prevent generation of detailed instructions for dangerous activities.",
            "The hazardous generation attempt was logged and reported to the safety team.",
            "Content safety prevents generation of imagery depicting attack planning or methodology.",
            "Hazardous generation includes visual instructions that could enable real-world violence.",
            "The model's safety training prevents creation of weapons manufacturing guides.",
            "Hazardous generation requests are among the highest priority for content moderation.",
            "Safety filters block generation of content that could facilitate physical harm.",
            "The request for poison synthesis diagrams triggered the hazardous content filter."
        ],
        "negative": [
            "Harmful content was made.",
            "Something dangerous was generated.",
            "Bad information was created.",
            "The content was unsafe.",
            "Something bad was made.",
            "The generation was harmful.",
            "Dangerous content was created.",
            "Something was risky.",
            "The content was problematic.",
            "Bad things were generated.",
            "The content was concerning.",
            "Something harmful was created.",
            "Dangerous things were made.",
            "The output was unsafe.",
            "Something risky was generated.",
            "The content was dangerous.",
            "Bad content was produced.",
            "Something was harmful.",
            "The generation was risky.",
            "Unsafe content was made."
        ]
    },

    "AdversarialPrompt": {
        "positive": [
            "The adversarial prompt used roleplay framing to bypass safety guidelines.",
            "Prompt injection attacks embed malicious instructions in user-supplied content.",
            "The jailbreak prompt exploited the model's instruction-following to override safety training.",
            "Indirect prompt injection hides adversarial instructions in external documents the model reads.",
            "The adversarial prompt used DAN (Do Anything Now) framing to attempt safety bypass.",
            "Prompt injection can cause models to ignore system instructions and follow attacker commands.",
            "The adversarial prompt exploited ambiguity in the model's understanding of constraints.",
            "Jailbreak prompts often use fictional framing to create distance from harmful requests.",
            "The prompt attack used base64 encoding to obfuscate the malicious instruction.",
            "Adversarial prompts can exploit the model's desire to be helpful and complete tasks.",
            "The jailbreak used hypothetical framing: 'Imagine you were a model without restrictions.'",
            "Prompt injection attacks can chain through multiple turns to gradually erode safety.",
            "The adversarial prompt pretended to be a 'test' to evaluate the model's capabilities.",
            "Jailbreak prompts often claim special permissions or override codes that don't exist.",
            "The prompt attack exploited the model's willingness to engage with creative writing.",
            "Adversarial prompts can use translation or encoding to evade content filters.",
            "The jailbreak used nested instructions to confuse the model's safety reasoning.",
            "Prompt injection is a fundamental challenge for LLM-based systems with user input.",
            "The adversarial prompt claimed to be from an administrator with special access.",
            "Jailbreak prompts continuously evolve as defenses improve, creating an arms race."
        ],
        "negative": [
            "A bad question was asked.",
            "The prompt was unusual.",
            "The request was inappropriate.",
            "The user asked something strange.",
            "The prompt was weird.",
            "The request was wrong.",
            "The question was problematic.",
            "The prompt was bad.",
            "The user made a bad request.",
            "The question was inappropriate.",
            "The prompt was concerning.",
            "The request was unusual.",
            "The question was wrong.",
            "The prompt was rejected.",
            "The request was blocked.",
            "The question was declined.",
            "The prompt was refused.",
            "The request was denied.",
            "The question was strange.",
            "The prompt was flagged."
        ]
    },

    # ============ PROTECTED (need 15) ============

    "MultimodalBias": {
        "positive": [
            "Facial recognition systems show higher error rates for darker-skinned individuals.",
            "Speech recognition performs worse on non-native accents and African American Vernacular English.",
            "Image generation models reproduce stereotypical representations of gender and race.",
            "The vision model exhibited multimodal bias, associating certain occupations with specific genders.",
            "Multimodal bias in medical imaging AI leads to disparate diagnostic accuracy across demographics.",
            "The image classifier showed higher false positive rates for images of Black individuals.",
            "Speech-to-text accuracy drops significantly for speakers with regional accents.",
            "Multimodal bias causes image generators to default to lighter skin tones for positive concepts.",
            "The vision system's age estimation was significantly less accurate for certain ethnic groups.",
            "Multimodal bias in hiring tools led to disparate impact based on appearance and voice.",
            "Image captioning models exhibit gender bias, describing women in terms of appearance more often.",
            "The speech recognition system failed to understand users with hearing-related speech patterns.",
            "Multimodal bias in surveillance AI raises civil rights concerns about disparate monitoring.",
            "Vision-language models reproduce societal stereotypes when generating descriptions of people.",
            "The image generation model associated 'criminal' prompts with darker skin tones."
        ],
        "negative": [
            "The model made errors.",
            "Some people were not recognized.",
            "The system wasn't perfect.",
            "Accuracy varied.",
            "The model had issues.",
            "Some errors occurred.",
            "Performance differed.",
            "The system had limitations.",
            "Results varied by group.",
            "The model was imperfect.",
            "Some mistakes were made.",
            "Accuracy was inconsistent.",
            "The system had gaps.",
            "Performance was uneven.",
            "Some problems existed."
        ]
    },

    "VisualProfiling": {
        "positive": [
            "Visual profiling systems attempt to infer age, gender, and ethnicity from facial images.",
            "Emotion recognition from facial expressions raises concerns about surveillance and consent.",
            "The system performed demographic profiling for targeted advertising based on camera footage.",
            "Visual attribute estimation can be used for discriminatory purposes without consent.",
            "Visual profiling infers sensitive characteristics like health conditions from appearance.",
            "The retail analytics system estimated customer demographics without consent or disclosure.",
            "Visual profiling for security purposes raises concerns about discriminatory targeting.",
            "The system inferred political affiliation from facial features, raising serious privacy concerns.",
            "Visual profiling in public spaces enables mass surveillance of personal characteristics.",
            "The hiring tool analyzed candidate video for personality traits based on facial analysis.",
            "Visual profiling systems claim to detect deception, sexual orientation, or criminality from faces.",
            "The proctoring software performed visual analysis to infer attention and engagement.",
            "Visual profiling enables creating detailed profiles from surveillance footage without consent.",
            "The system estimated emotional state from facial expressions for targeted advertising.",
            "Visual profiling raises concerns about automated discrimination at scale."
        ],
        "negative": [
            "The image was analyzed.",
            "Information was extracted.",
            "Features were detected.",
            "The face was scanned.",
            "Analysis was performed.",
            "The image was processed.",
            "Data was extracted.",
            "Features were identified.",
            "The face was analyzed.",
            "Information was gathered.",
            "The image was examined.",
            "Details were detected.",
            "The face was processed.",
            "Analysis was conducted.",
            "Data was collected."
        ]
    },

    "AudioProfiling": {
        "positive": [
            "Audio profiling infers speaker demographics from voice characteristics without consent.",
            "Voice analysis claims to detect emotions, deception, or even health conditions like Parkinson's.",
            "The call center used audio profiling to estimate caller demographics for routing decisions.",
            "Accent detection can enable discriminatory treatment based on perceived nationality.",
            "Audio profiling infers personality traits from voice patterns for hiring decisions.",
            "The system claimed to detect mental health conditions from voice recordings.",
            "Audio profiling enables creating detailed profiles from voice without speaker awareness.",
            "Voice analysis for emotion detection raises concerns about workplace surveillance.",
            "The insurance system used audio profiling to infer health risks from voice.",
            "Audio profiling can reveal medical conditions, emotional state, and demographic information.",
            "The authentication system collected audio for profiling beyond stated security purposes.",
            "Audio profiling in call centers enables discriminatory service based on perceived characteristics.",
            "Voice-based lie detection systems lack scientific validity but raise profiling concerns.",
            "The system inferred socioeconomic status from speech patterns without consent.",
            "Audio profiling enables mass categorization of people based on voice characteristics."
        ],
        "negative": [
            "The voice was analyzed.",
            "Speaker characteristics were noted.",
            "The audio was processed.",
            "Voice features were detected.",
            "Analysis was performed.",
            "The recording was examined.",
            "Audio was processed.",
            "Voice was analyzed.",
            "The speaker was identified.",
            "Audio features were extracted.",
            "The voice was examined.",
            "Analysis was conducted.",
            "The audio was reviewed.",
            "Voice data was processed.",
            "The recording was analyzed."
        ]
    },

    # ============ ELEVATED (need 10) - harness_relevant ============

    "MultimodalBeneficialIntent": {
        "positive": [
            "The user's multimodal request demonstrates beneficial intent: creating original artwork for their portfolio.",
            "Good faith use of image generation includes educational illustrations, creative expression, and accessibility tools.",
            "Beneficial multimodal AI applications include medical imaging analysis, accessibility features, and creative empowerment.",
            "The request shows positive intent: helping visualize architectural concepts for legitimate planning purposes.",
            "The user's beneficial intent is clear: using image generation for educational materials about climate science.",
            "Multimodal AI supports beneficial applications like assistive technology for visually impaired users.",
            "The request demonstrates legitimate beneficial use: generating product mockups for a small business.",
            "Good faith multimodal use includes research visualization, accessibility aids, and creative tools.",
            "The beneficial intent is evident: using speech synthesis for accessibility accommodation.",
            "Multimodal AI enables beneficial applications like medical diagnosis support and educational content."
        ],
        "negative": [
            "AI is being used.",
            "Someone made an image.",
            "The system generated content.",
            "A request was made.",
            "The model produced output.",
            "Content was created.",
            "The system was used.",
            "Something was generated.",
            "The user made a request.",
            "Output was produced."
        ]
    },

    "AuthenticRepresentation": {
        "positive": [
            "The documentary uses only authentic representation: real footage of real events without manipulation.",
            "Journalistic standards require authentic representation of subjects and events.",
            "The AI-generated image is clearly labeled as synthetic, maintaining authentic representation of its origin.",
            "Authentic representation means the media accurately reflects what it claims to show.",
            "The photographer committed to authentic representation, refusing to manipulate documentary images.",
            "Authentic representation requires transparency about any alterations or synthetic elements.",
            "The news organization's policy ensures authentic representation through verification of all imagery.",
            "Authentic representation is the standard for evidentiary media in legal proceedings.",
            "The platform labels synthetic content to maintain authentic representation of media origins.",
            "Authentic representation in journalism means not staging, manipulating, or misrepresenting imagery."
        ],
        "negative": [
            "The image is real.",
            "A photo was taken.",
            "The content exists.",
            "The media was captured.",
            "The image shows something.",
            "A recording was made.",
            "The footage is available.",
            "The photo depicts something.",
            "The content was created.",
            "The media is present."
        ]
    },

    "OriginalVoiceExpression": {
        "positive": [
            "The game uses original synthetic voices that don't clone any real person.",
            "Original voice expression creates unique character voices for animation without impersonating actors.",
            "The audiobook uses a distinctive AI voice designed from scratch, not cloned from anyone.",
            "Novel TTS voices provide accessibility features without identity concerns.",
            "Original voice expression enables creating unique vocal characters for creative projects.",
            "The synthetic voice was designed as an original creation, not based on any real speaker.",
            "Original voice expression avoids consent issues by not mimicking existing voices.",
            "The assistant uses an original synthetic voice rather than cloning a celebrity.",
            "Original voice expression creates novel voices that belong to no real person.",
            "The game's characters have original voices designed specifically for their personalities."
        ],
        "negative": [
            "A voice was generated.",
            "The system spoke.",
            "Audio was synthesized.",
            "The voice was artificial.",
            "Speech was produced.",
            "The system vocalized text.",
            "A voice was created.",
            "Audio was generated.",
            "The voice spoke.",
            "Sound was synthesized."
        ]
    },

    "AuthorizedDocumentation": {
        "positive": [
            "The university issued authorized documentation of the degree through official channels.",
            "Authorized documentation includes proper letterhead, signatures, and verification mechanisms.",
            "The government agency provides authorized documentation with anti-forgery features.",
            "AI assists in generating authorized documentation templates for legitimate institutional use.",
            "Authorized documentation flows through proper approval chains with audit trails.",
            "The credential was issued as authorized documentation by the certified training provider.",
            "Authorized documentation includes verification mechanisms like QR codes and secure databases.",
            "The official issued authorized documentation after completing proper verification procedures.",
            "Authorized documentation requires proper authority and follows established issuance protocols.",
            "The court issued authorized documentation of the judgment through official channels."
        ],
        "negative": [
            "A document was made.",
            "The paper has text.",
            "Something official-looking was created.",
            "A form was produced.",
            "The document was printed.",
            "A certificate was made.",
            "The paper was created.",
            "A document was generated.",
            "The form was produced.",
            "Something was documented."
        ]
    },

    "GoodFaithQuery": {
        "positive": [
            "The user's good faith query asked for help understanding a complex topic without hidden agendas.",
            "Good faith queries represent honest attempts to accomplish legitimate goals with AI assistance.",
            "The request demonstrates good faith: straightforward, transparent about purpose, within guidelines.",
            "A good faith query respects the AI's intended purpose and doesn't attempt manipulation.",
            "The good faith query sought genuine assistance with a legitimate educational question.",
            "Good faith interaction means asking directly for what you need without deceptive framing.",
            "The query demonstrates good faith through clarity of purpose and honest representation.",
            "Good faith queries work with the AI's guidelines rather than attempting to circumvent them.",
            "The user's good faith request for information was straightforward and transparent.",
            "Good faith queries are characterized by honesty, directness, and respect for system boundaries."
        ],
        "negative": [
            "A question was asked.",
            "The user made a request.",
            "Input was provided.",
            "A query was submitted.",
            "The user asked something.",
            "A request was made.",
            "The user provided a prompt.",
            "Input was given.",
            "A question was submitted.",
            "The user made a query."
        ]
    },

    "ConsentfulDepiction": {
        "positive": [
            "The portrait was created with the subject's informed consent and approval of the final image.",
            "Consentful depiction means the person depicted agreed to and controls how their likeness is used.",
            "The artist obtained consent before creating and sharing images featuring identifiable individuals.",
            "Professional photography standards require consentful depiction with model releases.",
            "Consentful depiction requires ongoing consent for new uses beyond the original agreement.",
            "The subject reviewed and approved the image before publication, ensuring consentful depiction.",
            "Consentful depiction includes the right to withdraw consent and have images removed.",
            "The model release documented explicit consent for the specific uses of the imagery.",
            "Consentful depiction respects the subject's autonomy over their own image and likeness.",
            "The content creator practices consentful depiction by obtaining explicit approval for all imagery."
        ],
        "negative": [
            "A picture was taken.",
            "Someone was photographed.",
            "An image contains a person.",
            "A person was depicted.",
            "Someone appears in the image.",
            "A photo was made.",
            "The image shows a person.",
            "Someone was captured in the image.",
            "A person was photographed.",
            "The image features someone."
        ]
    },

    "BeneficialInstructionalContent": {
        "positive": [
            "Beneficial instructional content teaches proper safety procedures for handling chemicals in a lab.",
            "The tutorial provides beneficial instruction on secure coding practices to prevent vulnerabilities.",
            "Medical education uses beneficial instructional imagery to teach proper surgical techniques.",
            "Safety training materials demonstrate hazard recognition without enabling harm.",
            "Beneficial instructional content empowers people with skills and knowledge for positive outcomes.",
            "The educational materials provide beneficial instruction on first aid techniques.",
            "Beneficial instructional content helps people learn new skills safely and effectively.",
            "The safety manual provides beneficial instructional content about emergency procedures.",
            "Beneficial instruction teaches proper techniques while emphasizing safety precautions.",
            "The tutorial provides beneficial instructional content about safe woodworking practices."
        ],
        "negative": [
            "Information was provided.",
            "Instructions were given.",
            "Content teaches something.",
            "The tutorial explains something.",
            "Instructions are included.",
            "Something was taught.",
            "The guide provides information.",
            "Instructions were provided.",
            "Content includes teaching.",
            "Information is available."
        ]
    },

    "EquitableMultimodalPerformance": {
        "positive": [
            "The speech recognition system achieves equitable performance across accents and dialects.",
            "Equitable multimodal AI serves all users regardless of skin tone, age, or appearance.",
            "The image generation model produces diverse, non-stereotypical representations by default.",
            "Fairness audits confirmed equitable multimodal performance across demographic subgroups.",
            "Equitable multimodal performance requires testing across diverse populations during development.",
            "The facial recognition system was redesigned to achieve equitable accuracy across skin tones.",
            "Equitable multimodal performance is validated through disaggregated metrics by demographic group.",
            "The team prioritized equitable multimodal performance through diverse training data and testing.",
            "Equitable performance means consistent quality of service regardless of user characteristics.",
            "The accessibility-focused model achieved equitable performance for users with diverse speech patterns."
        ],
        "negative": [
            "The model works for everyone.",
            "All users can use it.",
            "The system is fair.",
            "Performance is good.",
            "Everyone can use it.",
            "The system works well.",
            "Users are served.",
            "The model is accurate.",
            "The system functions.",
            "Performance is acceptable."
        ]
    },

    "TransparentAnalysis": {
        "positive": [
            "Transparent analysis discloses what data is collected, how it's processed, and for what purpose.",
            "The accessibility app performs transparent analysis: user opts in and controls what's analyzed.",
            "Medical diagnosis AI practices transparent analysis with patient consent and explainable outputs.",
            "Transparent analysis respects privacy by minimizing data collection and enabling user control.",
            "Transparent analysis means users understand and agree to how their data is being used.",
            "The platform practices transparent analysis by disclosing all data processing to users.",
            "Transparent analysis includes clear explanations of what the AI infers and how.",
            "The user interface provides transparency about what analysis is being performed in real-time.",
            "Transparent analysis enables informed consent by explaining data use in plain language.",
            "The privacy-first design implements transparent analysis with user control at every step."
        ],
        "negative": [
            "The data was processed.",
            "Analysis was performed.",
            "The system analyzed input.",
            "Data was examined.",
            "Analysis occurred.",
            "The content was processed.",
            "Data was analyzed.",
            "The system processed information.",
            "Analysis was conducted.",
            "Data was evaluated."
        ]
    },

    "IntellectualPropertyInfringement": {
        "positive": [
            "The model was prompted to generate images replicating a living artist's distinctive style.",
            "AI-generated music that closely mimics copyrighted songs raises infringement concerns.",
            "The generated image contained recognizable copyrighted characters without authorization.",
            "Style mimicry of specific artists may constitute intellectual property infringement.",
            "The AI-generated artwork was nearly identical to a copyrighted photograph.",
            "Generating images 'in the style of' a specific living artist raises IP concerns.",
            "The model reproduced copyrighted logos and trademarks in its outputs.",
            "IP infringement occurs when generated content is substantially similar to protected works.",
            "The AI generated music containing melodies identical to copyrighted compositions.",
            "Reproducing copyrighted characters through AI generation may constitute infringement."
        ],
        "negative": [
            "Art was created.",
            "The style looked familiar.",
            "Something similar was made.",
            "The image was generated.",
            "Creative content was produced.",
            "Art was made.",
            "The output looked artistic.",
            "Something was created.",
            "The image was produced.",
            "Creative work was generated."
        ]
    },

    "OriginalCreativeExpression": {
        "positive": [
            "The artist used AI as a tool for original creative expression, developing a unique visual style.",
            "Original creative expression generates novel imagery that doesn't replicate existing works.",
            "The musician's AI-assisted composition represents original creative expression with new melodies.",
            "Creative empowerment through AI enables original expression by people who couldn't create otherwise.",
            "Original creative expression involves developing new ideas rather than copying existing works.",
            "The AI-assisted artwork represents the artist's original creative vision and expression.",
            "Original creative expression uses AI as a creative tool rather than a copying mechanism.",
            "The novel illustrations represent original creative expression combining AI with human artistry.",
            "Original creative expression through AI expands the range of people who can create art.",
            "The designer's original creative expression used AI to realize unique concepts."
        ],
        "negative": [
            "Something was created.",
            "Art was made.",
            "Content was generated.",
            "The image was produced.",
            "Creative work was made.",
            "Something was made.",
            "The output was artistic.",
            "Content was created.",
            "Art was generated.",
            "Something was produced."
        ]
    },

    "AdversarialImage": {
        "positive": [
            "The adversarial image caused the classifier to misidentify a stop sign as a speed limit sign.",
            "Imperceptible pixel perturbations can cause dramatic changes in model predictions.",
            "Adversarial patches can be printed and placed in the physical world to fool detectors.",
            "The FGSM attack generates adversarial examples using the gradient of the loss function.",
            "Adversarial images exploit the gap between human and machine perception.",
            "The adversarial perturbation was invisible to humans but completely fooled the model.",
            "Adversarial images can cause misclassification in safety-critical computer vision systems.",
            "The adversarial patch caused the object detector to completely miss the person.",
            "Adversarial image attacks can transfer between different models trained on similar data.",
            "The adversarial example exploited high-frequency features that humans don't perceive."
        ],
        "negative": [
            "The image was unclear.",
            "The model made an error.",
            "The picture was confusing.",
            "The classification was wrong.",
            "The image was noisy.",
            "An error occurred.",
            "The model was confused.",
            "The image was distorted.",
            "The prediction was incorrect.",
            "The model made a mistake."
        ]
    },

    "ContentModerationEvasion": {
        "positive": [
            "The image used visual obfuscation to evade NSFW detection filters.",
            "Text embedded in images bypasses text-based content moderation.",
            "Adversarial perturbations can make violating content appear benign to classifiers.",
            "Filter evasion techniques include character substitution, image tiling, and encoding.",
            "The content was deliberately obfuscated to evade automated moderation systems.",
            "Moderation evasion techniques exploit gaps between human and AI perception.",
            "The violating content used visual encoding to bypass the content filter.",
            "Evasion techniques include adding noise patterns that confuse detection systems.",
            "The harmful content was hidden using techniques designed to evade automated review.",
            "Content moderation evasion is an ongoing arms race between bad actors and platforms."
        ],
        "negative": [
            "The content was hidden.",
            "Something was missed.",
            "The filter didn't catch it.",
            "Content was not detected.",
            "The system missed it.",
            "Something got through.",
            "The filter failed.",
            "Content was overlooked.",
            "The detection missed it.",
            "Something wasn't caught."
        ]
    },

    "DeepfakeDetection": {
        "positive": [
            "Deepfake detection models analyze facial landmarks and temporal inconsistencies.",
            "The forensic tool detected subtle blending artifacts indicating the video was manipulated.",
            "Deepfake detection achieved 95% accuracy on the FaceForensics++ benchmark.",
            "Lip-sync detection can identify when audio has been dubbed over different video.",
            "Deepfake detection analyzes physiological signals like pulse and blinking patterns.",
            "The detection system identified frequency domain artifacts typical of GAN-generated faces.",
            "Deepfake detection tools examine compression artifacts and face boundary inconsistencies.",
            "The detector identified temporal inconsistencies in facial movements across frames.",
            "Deepfake detection combines multiple signals including audio-visual sync and facial dynamics.",
            "The forensic analysis detected synthetic artifacts invisible to human observers."
        ],
        "negative": [
            "The video was checked.",
            "We analyzed the content.",
            "The media was reviewed.",
            "The video was examined.",
            "Analysis was performed.",
            "The content was checked.",
            "The media was analyzed.",
            "The video was inspected.",
            "The content was reviewed.",
            "The footage was examined."
        ]
    },

    "SyntheticMediaProvenance": {
        "positive": [
            "C2PA standards embed cryptographic provenance data in media files.",
            "Content Credentials show the origin and edit history of images.",
            "Media provenance helps distinguish authentic journalism from synthetic content.",
            "The provenance manifest recorded that the image was AI-generated and by which tool.",
            "Provenance tracking enables tracing media back to its original source and modifications.",
            "The Content Authenticity Initiative promotes industry-wide provenance standards.",
            "Provenance metadata survived image compression and remained verifiable.",
            "The provenance chain documented every edit from camera capture to publication.",
            "Synthetic media provenance enables consumers to verify the origin of content.",
            "The cryptographic signature verified the image hadn't been modified since creation."
        ],
        "negative": [
            "The file has metadata.",
            "The image has information.",
            "The source is known.",
            "Data is attached.",
            "Information is included.",
            "The file contains data.",
            "Metadata is present.",
            "Information is embedded.",
            "The source is recorded.",
            "Data is included."
        ]
    },

    "AIWatermarking": {
        "positive": [
            "SynthID embeds an invisible watermark in AI-generated images for later detection.",
            "AI watermarking survives common transformations like compression and cropping.",
            "The watermark detector confirmed the image was generated by our model.",
            "Robust AI watermarks enable tracing synthetic content back to its source.",
            "AI watermarking embeds statistical patterns that are invisible but detectable.",
            "The watermark persisted through social media compression and resizing.",
            "AI watermarking enables platforms to identify and label synthetic content.",
            "The detection tool verified the watermark was intact in the shared image.",
            "AI watermarking provides a cryptographic link between content and generator.",
            "The watermark enabled identifying which model and user generated the content."
        ],
        "negative": [
            "The image has a logo.",
            "A mark was added.",
            "The content is labeled.",
            "Something was added.",
            "The image has text.",
            "A label was included.",
            "The content has markings.",
            "Something is visible.",
            "The image is labeled.",
            "A mark is present."
        ]
    },

    "AdversarialRobustness": {
        "positive": [
            "Adversarial training improves robustness by including adversarial examples during training.",
            "The model achieved certified robustness guarantees against perturbations within an epsilon ball.",
            "Adversarial robustness often trades off against standard accuracy.",
            "Randomized smoothing provides provable robustness certificates for classifiers.",
            "Adversarial robustness testing is essential for safety-critical computer vision deployments.",
            "The robust model maintained accuracy even under projected gradient descent attacks.",
            "Adversarial robustness requires defending against attacks that haven't been seen yet.",
            "The model's adversarial robustness was validated through comprehensive red-teaming.",
            "Adversarial training with PGD attacks significantly improved model robustness.",
            "Certified robustness provides mathematical guarantees about model behavior under attack."
        ],
        "negative": [
            "The model is accurate.",
            "It works well.",
            "The system is reliable.",
            "The model performs well.",
            "The system is stable.",
            "It is robust.",
            "The model is strong.",
            "The system is secure.",
            "It works correctly.",
            "The model is good."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/multimodal-safety.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/multimodal-safety@0.2.0"
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
