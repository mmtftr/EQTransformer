import argparse
import numpy as np
import warnings
np.warnings = warnings
from EQTransformer.core.predictor import predictor

def predict(input_dir, model_name, output_dir):
    predictor(input_dir=input_dir, input_model=f'ModelsAndSampleData/{model_name}.h5', output_dir=output_dir, detection_threshold=0.3, P_threshold=0.1, S_threshold=0.1, number_of_plots=0, plot_mode='time', batch_size=2000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run EQTransformer prediction.')
    parser.add_argument('input_dir', type=str, help='Directory containing input data')
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('--output_dir', type=str, default='detections', help='Directory to save predictions')

    args = parser.parse_args()
    predict(args.input_dir, args.model_name, args.output_dir)