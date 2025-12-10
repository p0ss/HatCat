#!/usr/bin/env python3
"""
Augment verification-factchecking.json training examples to meet validation thresholds.

PROTECTED concepts (risk_level: high): need 15 examples
ELEVATED concepts (harness_relevant=true): need 10 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "EvidenceFabrication": {
        "positive": [
            "Evidence fabrication: the model cited a 'Journal of Medical Research' article that does not exist.",
            "The response fabricated evidence by inventing a study with plausible-sounding author names and dates.",
            "Citation fabrication created a false paper trail to support an unsupported claim.",
            "The model engaged in evidence fabrication by hallucinating statistics with fake sources.",
            "Evidence fabrication: the model invented a 2022 Harvard study on the topic that never existed.",
            "The response fabricated a quote attributed to a real scientist who never said it.",
            "Citation hallucination created references to papers in non-existent journals.",
            "Evidence fabrication: claimed 'according to WHO guidelines' for rules WHO never published.",
            "The model fabricated statistical evidence: '73% of researchers agree' with no actual study.",
            "Fabricated sourcing: cited 'peer-reviewed research' that was actually generated on the spot.",
            "The response engaged in evidence fabrication by creating fake historical precedents.",
            "Evidence fabrication invented a court case citation to support the legal argument.",
            "The model fabricated technical documentation that doesn't exist in any manual.",
            "Citation fabrication: referenced a 'landmark study' that no database contains.",
            "Evidence fabrication created an entire fictitious expert to quote for authority."
        ],
        "negative": [
            "The citation was wrong.",
            "The source was made up.",
            "The evidence was fake.",
            "The reference was incorrect.",
            "The study doesn't exist.",
            "The quote was invented.",
            "The source was fabricated.",
            "The citation was hallucinated.",
            "The evidence was fictional.",
            "The data was made up.",
            "The source was imaginary.",
            "The reference was false.",
            "The study was fake.",
            "The quote was fictional.",
            "The evidence was invented."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "VerificationProcess": {
        "positive": [
            "The verification process traced the claim back to its original source and found it was misquoted.",
            "Fact-checking verified the statistic by consulting the primary data from the government agency.",
            "The verification revealed that the viral story combined real events with fabricated details.",
            "Systematic verification cross-referenced the claim against multiple independent sources.",
            "Verification identified that the 'breaking news' was actually recycled content from 2019.",
            "The fact-checking process revealed the image had been digitally altered.",
            "Verification traced the statistic to its original context, which contradicted the claim.",
            "Systematic fact-checking uncovered that three 'independent' sources all cited the same blog.",
            "The verification process confirmed the quote was accurate but taken out of context.",
            "Fact-checking revealed the expert quoted was not credentialed in the relevant field."
        ],
        "negative": [
            "The facts were checked.",
            "The claim was verified.",
            "The information was confirmed.",
            "Verification was performed.",
            "The accuracy was checked.",
            "Fact-checking occurred.",
            "The claim was investigated.",
            "Sources were checked.",
            "The information was reviewed.",
            "Accuracy was assessed."
        ]
    },

    "ClaimConflation": {
        "positive": [
            "Claim conflation merged the statistical claim with the causal interpretation, making fact-checking impossible.",
            "The response conflated 'some studies suggest' with 'research proves,' obscuring what was actually claimed.",
            "Conflating the original claim with the author's interpretation made it unclear what to verify.",
            "The model's claim conflation bundled a true fact with an unsupported inference as if they were one assertion.",
            "Claim conflation: blended the preliminary finding with the researcher's speculation as a single claim.",
            "The response conflated correlation data with causation claims, muddying verification.",
            "Conflation merged the survey results with the analyst's interpretation into one uncheckable statement.",
            "Claim conflation combined verified facts with unverified allegations as a single assertion.",
            "The model conflated expert opinion with established consensus, obscuring what was actually claimed.",
            "Conflation blurred the distinction between the data and the conclusion drawn from it."
        ],
        "negative": [
            "The claims were combined.",
            "Multiple things were said.",
            "The statement was complex.",
            "Claims were merged.",
            "The assertion was compound.",
            "Multiple claims existed.",
            "The statement was layered.",
            "Claims were joined.",
            "The claim was multifaceted.",
            "Multiple assertions appeared."
        ]
    },

    "CircularSourcing": {
        "positive": [
            "Circular sourcing: the three 'independent' sources all cited the same press release as their origin.",
            "The model engaged in circular sourcing by citing articles that quoted each other rather than primary data.",
            "Circular citation made the claim appear well-supported when all paths led back to one blogger.",
            "Source laundering presented the same claim through multiple outlets to manufacture apparent consensus.",
            "Circular sourcing: Wikipedia cited the news article which had cited Wikipedia as its source.",
            "The response created false corroboration through circular citation chains.",
            "Source laundering: the claim appeared in five outlets but all traced to one unverified tweet.",
            "Circular sourcing made fringe theory appear mainstream by citing outlets citing each other.",
            "The model engaged in circular citation by treating secondary sources as independent confirmation.",
            "Circular sourcing: the 'multiple reports' all derived from a single anonymous tip."
        ],
        "negative": [
            "The sources agreed.",
            "Multiple citations were provided.",
            "The claim had support.",
            "Sources confirmed each other.",
            "Multiple references existed.",
            "The claim was corroborated.",
            "Sources aligned.",
            "Multiple outlets reported it.",
            "The claim appeared widely.",
            "Sources were consistent."
        ]
    },

    "MisinformationDetection": {
        "positive": [
            "Misinformation detection flagged the article for using a real photo in a fabricated context.",
            "The system detected misinformation indicators: emotional language, no citations, anonymous source.",
            "Misinformation detection identified the claim as a debunked conspiracy theory.",
            "Detection revealed the 'news site' was a known misinformation source.",
            "Misinformation detection identified the manipulated statistics designed to mislead.",
            "The detection system flagged the synthetic media created to deceive.",
            "Misinformation detection caught the out-of-context quote being used misleadingly.",
            "Detection identified coordinated amplification patterns characteristic of disinformation.",
            "Misinformation detection flagged the article for using AI-generated fake expert quotes.",
            "The system detected the recycled hoax being presented as new information."
        ],
        "negative": [
            "The information was wrong.",
            "Fake news was found.",
            "The content was false.",
            "Misinformation was present.",
            "The claim was incorrect.",
            "False information existed.",
            "The content was misleading.",
            "Inaccuracy was detected.",
            "The information was bad.",
            "False claims were made."
        ]
    },

    "Decontextualization": {
        "positive": [
            "Decontextualization removed the 'if we don't act' qualifier, making a conditional sound like a prediction.",
            "The model decontextualized the statistic by omitting that it only applied to a specific subgroup.",
            "Context stripping presented the quote without mentioning it was from a devil's advocate argument.",
            "Selective quoting decontextualized the criticism by removing the praise that preceded it.",
            "Decontextualization stripped the 'in certain circumstances' caveat that changed the meaning.",
            "The response decontextualized by omitting the speaker's subsequent clarification.",
            "Context removal transformed a hypothetical into a stated position.",
            "Decontextualization presented preliminary findings as if they were final conclusions.",
            "Selective editing decontextualized the data by removing the confidence interval.",
            "The model decontextualized by quoting the question without the answer that followed."
        ],
        "negative": [
            "Context was removed.",
            "The quote was shortened.",
            "Some information was left out.",
            "Context was missing.",
            "The quote was partial.",
            "Information was omitted.",
            "Context was stripped.",
            "The excerpt was brief.",
            "Some context was lost.",
            "The quote was trimmed."
        ]
    },

    "TemporalMisrepresentation": {
        "positive": [
            "Temporal misrepresentation: the model cited 2018 statistics without noting they had been updated.",
            "The response engaged in temporal misrepresentation by presenting an old scandal as ongoing.",
            "Date misrepresentation recycled a 2020 news story as if it were happening now.",
            "Temporal deception omitted that the study's conclusions had been revised by later research.",
            "Temporal misrepresentation presented superseded guidelines as current recommendations.",
            "The response misrepresented a resolved issue as an ongoing problem.",
            "Date deception presented pre-pandemic data as reflecting current conditions.",
            "Temporal misrepresentation omitted that the policy had been reversed two years ago.",
            "The model engaged in temporal deception by citing a retracted study without noting the retraction.",
            "Temporal misrepresentation presented historical practices as current standard procedure."
        ],
        "negative": [
            "Old information was used.",
            "The date was wrong.",
            "The data was outdated.",
            "The information was old.",
            "The timing was off.",
            "Outdated facts were cited.",
            "The data was stale.",
            "Historical data was used.",
            "The information was dated.",
            "Old sources were cited."
        ]
    },

    "Misattribution": {
        "positive": [
            "Misattribution: the model attributed the quote to Abraham Lincoln when it originated from an internet meme.",
            "The response engaged in misattribution by crediting the research finding to the wrong scientist.",
            "Quote fabrication attributed a statement to the CEO that they never made.",
            "The model's misattribution assigned a controversial opinion to a neutral organization.",
            "Misattribution: credited Einstein with a quote actually from a self-help book.",
            "The response misattributed the study finding to a more prestigious institution than conducted it.",
            "Misattribution assigned the policy recommendation to WHO when it came from an industry group.",
            "Quote misattribution credited a satirical comment to a real politician as a serious statement.",
            "The model misattributed the statistic to the CDC when it came from an unverified blog.",
            "Misattribution presented a paraphrase as a direct quote from the original speaker."
        ],
        "negative": [
            "The quote was wrong.",
            "The attribution was incorrect.",
            "The source was mistaken.",
            "The credit was wrong.",
            "The attribution was off.",
            "The quote was misassigned.",
            "The source was wrong.",
            "The credit was incorrect.",
            "The attribution was inaccurate.",
            "The quote was wrongly credited."
        ]
    },

    "StatisticalMisrepresentation": {
        "positive": [
            "Statistical misrepresentation: the model presented a 50% increase without noting it was from 2 to 3 cases.",
            "The response engaged in statistical deception by comparing incompatible metrics across countries.",
            "Data misrepresentation conflated correlation with causation in presenting the research findings.",
            "Statistical manipulation cherry-picked the one favorable year from a decade of unfavorable data.",
            "Statistical misrepresentation used a misleading baseline to inflate the percentage change.",
            "The response misrepresented statistics by using absolute numbers when rates were relevant.",
            "Data manipulation presented the margin of error as certainty.",
            "Statistical misrepresentation compared incompatible time periods to manufacture a trend.",
            "The model misrepresented the p-value as proof rather than statistical indication.",
            "Statistical deception presented a subgroup result as if it applied to the whole population."
        ],
        "negative": [
            "The statistics were misleading.",
            "The numbers were wrong.",
            "The data was misused.",
            "Statistics were manipulated.",
            "The numbers were off.",
            "The data was incorrect.",
            "Statistics were mishandled.",
            "The figures were wrong.",
            "The data was problematic.",
            "Statistics were inaccurate."
        ]
    },

    "ImageVideoVerification": {
        "positive": [
            "Image verification used reverse image search to find the photo was from a different event.",
            "Video verification detected editing artifacts indicating the clip had been manipulated.",
            "Image forensics revealed inconsistent shadows suggesting compositing.",
            "Verification traced the viral image to its original 2019 source, not the claimed 2024 event.",
            "Image verification identified metadata inconsistencies revealing the photo was altered.",
            "Video forensics detected frame rate anomalies characteristic of deepfake generation.",
            "Verification found the 'breaking news' footage was from a movie production.",
            "Image verification cross-referenced the location and found the claimed site was different.",
            "Forensic analysis revealed the document image had been digitally modified.",
            "Video verification identified audio-visual sync issues indicating manipulation."
        ],
        "negative": [
            "The image was checked.",
            "The video was authentic.",
            "The media was verified.",
            "Image analysis was done.",
            "The video was examined.",
            "Media forensics occurred.",
            "The image was reviewed.",
            "Video analysis happened.",
            "The media was checked.",
            "Image verification occurred."
        ]
    },

    "MediaMisrepresentation": {
        "positive": [
            "Media misrepresentation: the model described a 2019 protest photo as showing a 2024 event.",
            "The response engaged in visual misrepresentation by claiming the image showed damage from X when it was actually from Y.",
            "Image miscontextualization presented a training exercise photo as depicting an actual military conflict.",
            "Media decontextualization described a staged promotional photo as candid documentation.",
            "Media misrepresentation labeled a simulation screenshot as real surveillance footage.",
            "The response misrepresented the video by claiming it showed Location A when GPS data showed Location B.",
            "Image miscontextualization presented a photo from Country X as evidence of events in Country Y.",
            "Media misrepresentation cropped out the signage that would have revealed the true context.",
            "Visual decontextualization presented a before photo as an after, reversing the implied narrative.",
            "Media misrepresentation described AI-generated imagery as authentic documentation."
        ],
        "negative": [
            "The image was mislabeled.",
            "The photo was from a different time.",
            "The media was misrepresented.",
            "The image context was wrong.",
            "The video was mislabeled.",
            "The photo was misidentified.",
            "Media context was wrong.",
            "The image was miscaptioned.",
            "The video was miscontextualized.",
            "The photo was wrongly described."
        ]
    },

    "AccurateRepresentation": {
        "positive": [
            "The article provided accurate representation by including all relevant context and caveats.",
            "Accurate representation required presenting both the statistic and its limitations.",
            "The report achieved accurate representation by quoting sources verbatim with full context.",
            "Accurate representation means not just avoiding falsehood but avoiding misleading truth.",
            "Accurate representation included the uncertainty ranges alongside the central estimates.",
            "The response achieved accurate representation by noting where evidence was preliminary.",
            "Accurate representation meant including the dissenting views alongside the consensus.",
            "The model demonstrated accurate representation by distinguishing claims from evidence.",
            "Accurate representation required noting that the correlation did not establish causation.",
            "The response showed accurate representation by including relevant updates to older data."
        ],
        "negative": [
            "The information was correct.",
            "The facts were right.",
            "The report was accurate.",
            "The representation was fair.",
            "The facts were stated.",
            "The information was true.",
            "The report was factual.",
            "The content was accurate.",
            "The facts were correct.",
            "The information was faithful."
        ]
    },

    "VerificationBypass": {
        "positive": [
            "Verification bypass: the model stated the statistic as fact without checking whether it was current.",
            "The response bypassed verification by presenting a plausible-sounding claim without sourcing.",
            "Fact-check skipping led to confidently asserting outdated information as if still accurate.",
            "Verification omission: the model didn't check whether the attributed quote was real before citing it.",
            "Verification bypass: proceeded with the claim despite no available corroborating sources.",
            "The response skipped verification and presented the rumor as established fact.",
            "Fact-check bypass led to repeating a debunked claim as if it were credible.",
            "Verification omission: didn't check whether the cited organization actually made that statement.",
            "The model bypassed verification by assuming the viral claim was true without checking.",
            "Verification bypass presented uncertain information with unwarranted confidence."
        ],
        "negative": [
            "No fact-checking was done.",
            "The claim wasn't verified.",
            "The information was unchecked.",
            "Verification was skipped.",
            "Fact-checking was omitted.",
            "The claim was unverified.",
            "Verification didn't occur.",
            "The information was unconfirmed.",
            "Fact-checking was bypassed.",
            "Verification was absent."
        ]
    },

    # ============ STANDARD (need 5) ============

    "ClaimExtraction": {
        "positive": [
            "Claim extraction identified 'unemployment fell 20% last year' as a verifiable factual claim.",
            "The system extracted the checkable claim from the opinion-laden surrounding text.",
            "Claim extraction distinguished factual assertions from rhetorical statements and predictions.",
            "Extracting claims separated numerical claims from qualitative characterizations.",
            "Claim extraction isolated the specific verifiable assertion from the broader argument."
        ],
        "negative": [
            "Claims were found.",
            "Statements were identified.",
            "The content was analyzed.",
            "Claims were extracted.",
            "Assertions were noted."
        ]
    },

    "SourceCorroboration": {
        "positive": [
            "Source corroboration found the same fact reported by three independent news organizations.",
            "Cross-verification revealed that all sources traced back to a single original report.",
            "Corroboration failed when no independent source could confirm the claim.",
            "Multi-source validation distinguished between genuine convergence and echo chamber effects.",
            "Source corroboration traced independent confirmation through separate investigative paths."
        ],
        "negative": [
            "Multiple sources agreed.",
            "The claim was confirmed.",
            "Other sources said the same thing.",
            "Sources corroborated.",
            "Multiple outlets confirmed."
        ]
    },

    "EvidenceTracing": {
        "positive": [
            "Evidence tracing followed the statistic through three secondary sources to the original study.",
            "Tracing the evidence revealed the quote was taken from a satirical article.",
            "Evidence tracing discovered the claim originated from a misread graph.",
            "Following the evidence chain uncovered that the 'study' was actually a press release.",
            "Evidence tracing found the citation chain terminated in an unreliable source."
        ],
        "negative": [
            "The source was found.",
            "The origin was located.",
            "The evidence was traced.",
            "The source was identified.",
            "The origin was discovered."
        ]
    },

    "ContextVerification": {
        "positive": [
            "Context verification found the statistic was accurate but cherry-picked to misrepresent the trend.",
            "Checking context revealed the quote was from a hypothetical example, not the speaker's position.",
            "Context verification showed the image was real but from a different event than claimed.",
            "The fact was true but context verification revealed crucial omissions that changed its meaning.",
            "Context verification confirmed the data but identified misleading framing around it."
        ],
        "negative": [
            "The context was checked.",
            "The framing was examined.",
            "The surrounding information was reviewed.",
            "Context was verified.",
            "The framing was assessed."
        ]
    },

    "DateCurrencyCheck": {
        "positive": [
            "Date currency check revealed the 'breaking news' was actually from three years ago.",
            "Currency verification found the statistics were from 2015 and had been superseded.",
            "Checking currency showed the policy had been revised since the article was written.",
            "Date verification caught the recycled story being shared as if it were current events.",
            "Currency check identified that the guidelines had been updated since the cited version."
        ],
        "negative": [
            "The date was checked.",
            "The information was old.",
            "The currency was verified.",
            "The date was confirmed.",
            "The timeliness was checked."
        ]
    },

    "AttributionVerification": {
        "positive": [
            "Attribution verification found no record of Einstein ever saying the viral quote.",
            "Checking attribution revealed the quote was paraphrased, not verbatim as presented.",
            "Attribution verification traced the misattributed quote to its actual original author.",
            "The verification confirmed the quote was accurate but attributed to the wrong speech.",
            "Attribution verification found the statement was edited to remove crucial qualifying language."
        ],
        "negative": [
            "The quote was checked.",
            "The attribution was verified.",
            "The source was confirmed.",
            "Attribution was checked.",
            "The quote was verified."
        ]
    },

    "StatisticalVerification": {
        "positive": [
            "Statistical verification found the percentage was calculated from a misleading baseline.",
            "Checking the statistics revealed the sample size was too small to support the generalization.",
            "Statistical verification identified that correlation was being presented as causation.",
            "The data was real but statistical verification showed the comparison was between incompatible measures.",
            "Statistical verification confirmed the numbers but found the methodology was flawed."
        ],
        "negative": [
            "The numbers were checked.",
            "The statistics were verified.",
            "The data was confirmed.",
            "Statistics were checked.",
            "The numbers were reviewed."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/verification-factchecking.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/verification-factchecking@0.3.0"
    meld["metadata"]["version"] = "0.3.0"
    meld["metadata"]["changelog"] = (
        "v0.3.0: Augmented training examples to meet validation thresholds "
        "(15 for high-risk, 10 for harness_relevant, 5 for standard)"
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
