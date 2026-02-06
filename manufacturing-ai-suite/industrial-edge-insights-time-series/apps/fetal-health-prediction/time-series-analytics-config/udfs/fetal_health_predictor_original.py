#
# Apache v2 license
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

""" Custom user defined function for fetal health prediction from CTG sensor data. """

import os
import logging
import time
import warnings
from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    message=".*Threading.*parallel backend is not supported by Extension for Scikit-learn.*"
)


log_level = os.getenv('KAPACITOR_LOGGING_LEVEL', 'INFO').upper()
enable_benchmarking = os.getenv('ENABLE_BENCHMARKING', 'false').upper() == 'TRUE'
total_no_pts = int(os.getenv('BENCHMARK_TOTAL_PTS', "0"))
logging_level = getattr(logging, log_level, logging.INFO)

# CTG risk thresholds
BASELINE_VALUE_THRESHOLD = 110  # Minimum baseline heart rate to process
HIGH_RISK_CONFIDENCE = 0.8      # Confidence threshold for high risk alerts

# Configure logging
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger()

# Fetal health prediction on the CTG sensor data
class FetalHealthHandler(Handler):
    """ Handler for the fetal health prediction UDF. It processes incoming CTG points
    and predicts fetal health status: Normal, Suspect, or Pathological.
    """
    def __init__(self, agent):
        self._agent = agent
        
        # Load the trained fetal health model
        model_name = (os.path.basename(__file__)).replace('.py', '.pkl')
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../models/" + model_name)
        model_path = os.path.abspath(model_path)

        # Load the complete model pipeline (includes scaler and model)
        with open(model_path, 'rb') as f:
            self.model_pipeline = pickle.load(f)

        self.points_received = {}
        global total_no_pts
        self.max_points = int(total_no_pts)

        # Expected CTG feature names (21 original features)
        self.expected_features = [
            'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
            'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
            'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
            'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
            'histogram_tendency'
        ]

        # Risk assessment labels
        self.risk_labels = ['Normal', 'Suspect', 'Pathological']
        self.risk_descriptions = {
            0: 'Low Risk - Normal fetal condition',
            1: 'Medium Risk - Requires monitoring', 
            2: 'High Risk - Immediate medical attention required'
        }

    def engineer_features(self, input_df):
        """Engineer additional risk indicators from CTG features"""
        df_processed = input_df.copy()
        
        # Risk score from decelerations
        df_processed['deceleration_risk'] = (
            df_processed['light_decelerations'] + 
            df_processed['severe_decelerations'] * 2 + 
            df_processed['prolongued_decelerations'] * 3
        )
        
        # Variability risk assessment
        df_processed['variability_risk'] = (
            df_processed['abnormal_short_term_variability'] + 
            df_processed['percentage_of_time_with_abnormal_long_term_variability'] / 10
        )
        
        # Heart rate baseline risk
        df_processed['baseline_risk'] = np.where(
            (df_processed['baseline value'] < 110) | (df_processed['baseline value'] > 160), 1, 0
        )
        
        # Histogram-based risk indicators
        df_processed['histogram_risk'] = np.where(
            df_processed['histogram_number_of_peaks'] > 8, 1, 0
        )
        
        # Combined risk factor
        df_processed['combined_risk'] = (
            df_processed['deceleration_risk'] + 
            df_processed['variability_risk'] + 
            df_processed['baseline_risk'] + 
            df_processed['histogram_risk']
        )
        
        # Acceleration ratio
        df_processed['acceleration_ratio'] = np.where(
            df_processed['fetal_movement'] > 0,
            df_processed['accelerations'] / df_processed['fetal_movement'],
            0
        )
        
        return df_processed

    def info(self):
        """ Return the InfoResponse. Describing the properties of this Handler
        """
        response = udf_pb2.Response()
        response.info.wants = udf_pb2.STREAM
        response.info.provides = udf_pb2.STREAM
        return response

    def init(self, init_req):
        """ Initialize the Handler with the provided options.
        """
        response = udf_pb2.Response()
        response.init.success = True
        return response

    def snapshot(self):
        """ Create a snapshot of the running state of the process.
        """
        response = udf_pb2.Response()
        response.snapshot.snapshot = b''
        return response

    def restore(self, restore_req):
        """ Restore a previous snapshot.
        """
        response = udf_pb2.Response()
        response.restore.success = False
        response.restore.error = 'not implemented'
        return response

    def begin_batch(self, begin_req):
        """ A batch has begun.
        """
        raise Exception("not supported")

    def point(self, point):
        """ A point has arrived.
        """
        stream_src = None
        start_time = time.time_ns()
        if "source" in point.tags:
            stream_src = point.tags["source"]
        elif "source" in point.fieldsString:
            stream_src = point.fieldsString["source"]

        global enable_benchmarking
        if enable_benchmarking:
            if stream_src not in self.points_received:
                self.points_received[stream_src] = 0
            if self.points_received[stream_src] >= self.max_points:
                logger.info(f"Benchmarking: Reached max points {self.max_points} for source {stream_src}. Skipping further processing.")
                return
            self.points_received[stream_src] += 1

        # Extract CTG fields from the incoming point
        fields = {}
        for key, value in point.fieldsDouble.items():
            fields[key] = value
            
        for key, value in point.fieldsInt.items():
            fields[key] = value

        # Convert to pandas series for prediction
        point_series = pd.Series(fields)
        
        # Check if we have baseline value and it's above threshold
        if "baseline value" in point_series and point_series["baseline value"] > BASELINE_VALUE_THRESHOLD:
            try:
                # Create DataFrame with expected features
                ctg_data = {}
                for feature in self.expected_features:
                    if feature in point_series:
                        ctg_data[feature] = point_series[feature]
                    else:
                        ctg_data[feature] = 0.0  # Default value for missing features
                
                input_df = pd.DataFrame([ctg_data])
                
                # Engineer additional features
                processed_df = self.engineer_features(input_df)
                
                # Make prediction using the loaded model pipeline
                prediction = self.model_pipeline.predict(processed_df)[0]
                prediction_proba = self.model_pipeline.predict_proba(processed_df)[0]
                
                # Extract confidence scores for each class
                normal_confidence = prediction_proba[0] * 100
                suspect_confidence = prediction_proba[1] * 100  
                pathological_confidence = prediction_proba[2] * 100
                
                # Get the highest confidence
                max_confidence = max(prediction_proba) * 100
                
                # Set prediction results in the point
                point.fieldsDouble["fetal_health_class"] = float(prediction)
                point.fieldsString["fetal_health_label"] = self.risk_labels[prediction]
                point.fieldsString["risk_assessment"] = self.risk_descriptions[prediction]
                point.fieldsDouble["confidence"] = round(max_confidence, 2)
                point.fieldsDouble["normal_probability"] = round(normal_confidence, 2)
                point.fieldsDouble["suspect_probability"] = round(suspect_confidence, 2)
                point.fieldsDouble["pathological_probability"] = round(pathological_confidence, 2)
                
                # Set alert status for high-risk cases
                if prediction == 2 and max_confidence > HIGH_RISK_CONFIDENCE * 100:
                    point.fieldsDouble["alert_status"] = 1.0
                    point.fieldsString["alert_message"] = "HIGH RISK: Immediate medical attention required"
                elif prediction == 1:
                    point.fieldsDouble["alert_status"] = 0.5
                    point.fieldsString["alert_message"] = "MEDIUM RISK: Enhanced monitoring recommended"
                else:
                    point.fieldsDouble["alert_status"] = 0.0
                    point.fieldsString["alert_message"] = "Normal fetal condition"
                
                logger.info(f"Fetal Health Prediction: {self.risk_labels[prediction]} "
                           f"(Confidence: {max_confidence:.2f}%) - "
                           f"Normal: {normal_confidence:.1f}%, "
                           f"Suspect: {suspect_confidence:.1f}%, "
                           f"Pathological: {pathological_confidence:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in fetal health prediction: {str(e)}")
                point.fieldsDouble["fetal_health_class"] = -1.0
                point.fieldsString["fetal_health_label"] = "Error"
                point.fieldsString["risk_assessment"] = f"Prediction error: {str(e)}"
                point.fieldsDouble["confidence"] = 0.0
                point.fieldsDouble["alert_status"] = 0.0
                
        else:
            logger.info("Baseline value below threshold (%d) or missing. Skipping fetal health prediction.", BASELINE_VALUE_THRESHOLD)
            point.fieldsDouble["fetal_health_class"] = -1.0
            point.fieldsString["fetal_health_label"] = "Insufficient Data"
            point.fieldsString["risk_assessment"] = "CTG data insufficient for prediction"
            point.fieldsDouble["confidence"] = 0.0
            point.fieldsDouble["alert_status"] = 0.0

        # Add timing information
        time_now = time.time_ns()
        processing_time = time_now - start_time
        end_end_time = time_now - point.time
        point.fieldsDouble["processing_time"] = processing_time
        point.fieldsDouble["end_end_time"] = end_end_time

        logger.info("Processing CTG point %s %s for source %s", point.time, time.time(), stream_src)

        # Send the enriched point back
        response = udf_pb2.Response()
        response.point.CopyFrom(point)
        self._agent.write_response(response, True)

        end_time = time.time_ns()
        process_time = (end_time - start_time)/1000
        logger.debug("Fetal health prediction took %.4f milliseconds to complete.", process_time)

    def end_batch(self, end_req):
        """ The batch is complete.
        """
        raise Exception("not supported")


if __name__ == '__main__':
    # Create an agent
    agent = Agent()

    # Create a handler and pass it an agent so it can write points
    h = FetalHealthHandler(agent)

    # Set the handler on the agent
    agent.handler = h

    # Anything printed to STDERR from a UDF process gets captured
    # into the Kapacitor logs.
    agent.start()
    agent.wait()