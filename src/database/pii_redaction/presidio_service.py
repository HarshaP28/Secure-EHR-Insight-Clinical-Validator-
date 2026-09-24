from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine


class ClinicalPIIRedactor:
    def __init__(self):
        print("Initializing Microsoft Presidio Zero-Trust Middleware...")

        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # -- FDE FIX 1: Custom SSN Pattern Recognizer
        # Catches any XXX-XX-XXXX format.
        ssn_pattern = Pattern(
            name="catch_all_ssn",
            regex=r"\d{3}-\d{2}-\d{4}",
            score=0.9
        )

        ssn_recognizer = PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[ssn_pattern]
        )

        self.analyzer.registry.add_recognizer(ssn_recognizer)

        # -- FDE FIX 2: Custom Deny-List for Medical Facilities
        # Forces Presidio to recognize specific hospital names
        # as organizations.

        hospital_recognizer = PatternRecognizer(
            supported_entity="ORGANIZATION",
            deny_list=[
                "Massachusetts General Hospital",
                "Mayo Clinic",
                "Cleveland Clinic"
            ],
            deny_list_score=1.0
        )

        self.analyzer.registry.add_recognizer(hospital_recognizer)

        # Entities that we want Presidio to detect.
        self.target_entities = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SSN",
            "LOCATION",
            "ORGANIZATION",
            "DATE_TIME"
        ]

    def redact_clinical_context(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        # Step 1: Analyze the text and find possible PII.
        analyzer_results = self.analyzer.analyze(
            text=raw_text,
            entities=self.target_entities,
            language="en",
            score_threshold=0.4
        )

        # Step 2: Replace/redact the detected PII.
        anonymized_result = self.anonymizer.anonymize(
            text=raw_text,
            analyzer_results=analyzer_results
        )

        # Step 3: Return the safe/redacted text.
        return anonymized_result.text


if __name__ == "__main__":
    redactor = ClinicalPIIRedactor()

    simulated_ehr_note = """
    Patient John Doe (SSN: 234-00-1234) was admitted to
    Massachusetts General Hospital on March 15th following
    a severe reaction to Furosemide. Wife Jane Doe can be reached
    at 415-555-0198 or jane.doe@email.com
    """

    print("\nRAW PHI FROM DATABASE:")
    print(simulated_ehr_note.strip())

    print("\nREDACTED OUTPUT (safe for LLM prompt):")

    safe_text = redactor.redact_clinical_context(simulated_ehr_note)

    print(safe_text.strip())