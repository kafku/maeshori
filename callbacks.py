# couding: utf-8

import os, time, json
from datetime import datetime
import urllib
from keras.callbacks import Callback


class WebHook(Callback):
    """
    Push notification through WebHook
    """

    def __init__(self, url, job_name=None):
        self.url = url
        if job_name is not None:
            self.job_name = job_name
        else:
            self.job_name = "job_" + os.uname()[1] + "_" + time.strftime("%Y%m%d%H%M")

        self.method = "POST"
        self.headers = {"Content-Type" : "application/json"}

    def on_response(self, response_body):
        pass

    def post_message(self, message):
        json_data = json.dumps(message).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=json_data, headers=self.headers, method=self.method)
        with urllib.request.urlopen(request) as response:
            response_body = response.read().decode("utf-8")
            self.on_response(response_body)

class IftttMakerWebHook(WebHook):
    """
    Push notification through IFTTT Maker WebHook
    """

    def __init__(self, url, job_name=None):
        self.start_time = None
        super(IftttMakerWebHook, self).__init__(url, job_name)

    def on_epoch_begin(self, epoch, logs=None):
        message = {
            "value1" : self.job_name,
            "value2" : "Epoch %d has begun."%(epoch + 1)
        }
        self.post_message(message)

    def on_epoch_end(self, epoch, logs=None):
        metrics_str = ""
        for k in self.params['metrics']:
            if k in logs:
                metrics_str += "%s: %0.4f/"%(k,logs[k])

        message = {
            "value1" : self.job_name,
            "value2" : "Epoch %d ended."%(epoch + 1),
            "value3" : metrics_str
        }
        self.post_message(message)

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        message = {
            "value1" : self.job_name,
            "value2" : "Training phase has started."
        }
        self.post_message(message)

    def on_train_end(self, logs=None):
        diff_time = datetime.now() - self.start_time
        duration_h = 24.0 * diff_time.days + diff_time.seconds / 3600
        message = {
            "value1" : self.job_name,
            "value2" : "Training phase ended.",
            "value3" : "Duration: %0.3f [h]"%duration_h
        }
        self.post_message(message)


class SlackWebHook(WebHook):
    pass
