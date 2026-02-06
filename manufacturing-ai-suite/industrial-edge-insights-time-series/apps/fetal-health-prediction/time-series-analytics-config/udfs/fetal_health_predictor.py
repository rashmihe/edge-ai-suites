#
# Apache v2 license
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

""" Simple test UDF for fetal health prediction from CTG sensor data. """

import os
import logging
import time
import warnings
from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
import random

warnings.filterwarnings("ignore")

log_level = os.getenv('KAPACITOR_LOGGING_LEVEL', 'INFO').upper()
enable_benchmarking = os.getenv('ENABLE_BENCHMARKING', 'false').upper() == 'TRUE'
total_no_pts = int(os.getenv('BENCHMARK_TOTAL_PTS', "0"))
logging_level = getattr(logging, log_level, logging.INFO)

# CTG risk thresholds
BASELINE_VALUE_THRESHOLD = 110  # Minimum baseline heart rate to process

# Configure logging
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger()

# Simple fetal health prediction on the CTG sensor data
class FetalHealthHandler(Handler):
    """ Handler for the fetal health prediction UDF. It processes incoming CTG points
    and provides simple rule-based fetal health classification.
    """
    def __init__(self, agent):
        self._agent = agent
        self.points_received = {}
        global total_no_pts
        self.max_points = int(total_no_pts)

        # Risk assessment labels
        self.risk_labels = ['Normal', 'Suspect', 'Pathological']
        self.risk_descriptions = {
            0: 'Low Risk - Normal fetal condition',
            1: 'Medium Risk - Requires monitoring', 
            2: 'High Risk - Immediate medical attention required'
        }

        logger.info("Fetal Health Predictor UDF initialized successfully")

    def simple_risk_assessment(self, baseline_value, accelerations, decelerations):
        """Simple rule-based risk assessment"""
        
        # Basic risk scoring
        risk_score = 0
        
        # Check baseline heart rate (more sensitive ranges)
        if baseline_value < 100 or baseline_value > 170:
            risk_score += 3  # Severe abnormality
        elif baseline_value < 110 or baseline_value > 160:
            risk_score += 2  # Moderate abnormality
        elif baseline_value < 115 or baseline_value > 155:
            risk_score += 1  # Mild abnormality
            
        # Check accelerations (should be present)
        if accelerations < 1:
            risk_score += 2  # Concerning
        elif accelerations < 2:
            risk_score += 1  # Borderline
            
        # Check for decelerations (should be minimal)
        if decelerations > 2:
            risk_score += 2  # Concerning
        elif decelerations > 0:
            risk_score += 1  # Any deceleration is notable
            
        # Add some randomness to simulate ML confidence
        confidence = max(0.6, min(0.95, 0.8 + random.uniform(-0.1, 0.1)))
        
        # More sensitive classification thresholds
        if risk_score >= 3:
            return 2, confidence  # Pathological (lowered from 4)
        elif risk_score >= 1:
            return 1, confidence  # Suspect (lowered from 2)
        else:
            return 0, confidence  # Normal
            
        # Debug logging
        logger.info(f"Risk Assessment: score={risk_score}, baseline={baseline_value}, acc={accelerations}, decel={decelerations}")

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

        # Check if we have baseline value and it's above threshold
        baseline_value = fields.get("baseline value", 120)
        accelerations = fields.get("accelerations", 2)
        decelerations = fields.get("light_decelerations", 0) + fields.get("severe_decelerations", 0)
        
        if baseline_value > BASELINE_VALUE_THRESHOLD:
            try:
                # Simple rule-based prediction
                prediction, confidence = self.simple_risk_assessment(baseline_value, accelerations, decelerations)
                
                # Set prediction results in the point
                point.fieldsDouble["fetal_health_class"] = float(prediction)
                point.fieldsString["fetal_health_label"] = self.risk_labels[prediction]
                point.fieldsString["risk_assessment"] = self.risk_descriptions[prediction]
                point.fieldsDouble["confidence"] = round(confidence * 100, 2)
                point.fieldsDouble["normal_probability"] = round(70.0 if prediction == 0 else 20.0, 2)
                point.fieldsDouble["suspect_probability"] = round(70.0 if prediction == 1 else 15.0, 2)
                point.fieldsDouble["pathological_probability"] = round(70.0 if prediction == 2 else 10.0, 2)
                
                # Calculate continuous alert status based on risk level and confidence
                if prediction == 2:  # Pathological (High Risk)
                    # Range: 0.7-1.0 based on confidence
                    alert_status = 0.7 + (confidence * 0.3)  # Maps confidence 0.0-1.0 to alert 0.7-1.0
                    if confidence > 0.7:
                        point.fieldsString["alert_message"] = f"HIGH RISK: Immediate attention required (Confidence: {confidence*100:.1f}%)"
                        logger.warning(f"HIGH RISK ALERT: confidence={confidence:.2f}, alert_status={alert_status:.2f}")
                    else:
                        point.fieldsString["alert_message"] = f"POTENTIAL HIGH RISK: Monitor closely (Confidence: {confidence*100:.1f}%)"
                        logger.info(f"POTENTIAL HIGH RISK: confidence={confidence:.2f}, alert_status={alert_status:.2f}")
                elif prediction == 1:  # Suspect (Medium Risk)
                    # Range: 0.3-0.7 based on confidence
                    alert_status = 0.3 + (confidence * 0.4)  # Maps confidence 0.0-1.0 to alert 0.3-0.7
                    if confidence > 0.6:
                        point.fieldsString["alert_message"] = f"MEDIUM RISK: Enhanced monitoring recommended (Confidence: {confidence*100:.1f}%)"
                        logger.warning(f"MEDIUM RISK ALERT: confidence={confidence:.2f}, alert_status={alert_status:.2f}")
                    else:
                        point.fieldsString["alert_message"] = f"MILD CONCERN: Continue monitoring (Confidence: {confidence*100:.1f}%)"
                        logger.info(f"MILD CONCERN: confidence={confidence:.2f}, alert_status={alert_status:.2f}")
                else:  # Normal (Low Risk)
                    # Range: 0.0-0.3 based on confidence (inverted - high confidence normal = low alert)
                    alert_status = 0.3 * (1.0 - confidence)  # Maps confidence 0.0-1.0 to alert 0.3-0.0
                    point.fieldsString["alert_message"] = f"Normal fetal condition (Confidence: {confidence*100:.1f}%)"
                
                point.fieldsDouble["alert_status"] = round(alert_status, 2)
                    
                # Always log alert status for debugging
                logger.info(f"Alert Status: {alert_status:.2f}, Message: {point.fieldsString['alert_message']}")
                f"Baseline: {baseline_value}, Accelerations: {accelerations}, Decelerations: {decelerations}"
                
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

        logger.debug("Processing CTG point %s %s for source %s", point.time, time.time(), stream_src)

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