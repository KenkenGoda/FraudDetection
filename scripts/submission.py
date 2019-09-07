from .db import LocalFile


class Submission:
    def __init__(self, config):
        self.config = config
        self.target_name = config.target_name
        self.save_path = config.submission_path

    def save(self, prediction):
        db = LocalFile(self.config)
        submission = db.get_submission()
        print("Succeeded in loading sample_submission.csv")
        submission[self.target_name] = prediction
        submission.to_csv(self.save_path, index=False)
        print("Succeeded in saving results for submission")
