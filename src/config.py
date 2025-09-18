INTENTS = ["billing", "kyc", "fraud_report", "sanctions", "tech_support", "general"]
LABEL2ID = {lab: i for i, lab in enumerate(INTENTS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 384