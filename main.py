import sys
import os



# ✅ Now we can import datascience
from src.datascience.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        print("🚀 Starting GAN Art Generation Training Pipeline...")
        pipeline = TrainingPipeline()
        pipeline.run()
        print("✅ Training Pipeline finished successfully.")
    except Exception as e:
        print("❌ Error while running pipeline:", str(e))


if __name__ == "__main__":
    main()
