# Project Write-Up

## Submission Details
MODEL NAME  : ssd_mobilenet_v2_coco_2018_03_29
MODEL LINK  : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
COMMANDS    : tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
              python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel


## Explaining Custom Layers

Layers apart from the known list of layers come under custom layers. Incase we use custom layers in our project we have to add extensions to both the model optimizer as well as the inference engine so that it can be handled during the time we do inference by our model.
My model involves dealing with no custom layers however there is support added to check if any layer is unsupported as custom layers behave differently which cant be handled by the toolkit yet.


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...
PRE   : Without changing any parameters the accuracy was just fine.
POST  : Post conversion i felt an increase in the accuracy I was getting more probability values than before.

The size of the model pre- and post-conversion was...
PRE   : 66.4 MB
POST  : 64.4 MB

The inference time of the model pre- and post-conversion was...
PRE   : 110.67 ms
POST  : 1.29 ms

The cpu overhead of the model pre- and post-conversion was...
PRE   : 67 %
POST  : 27 %

The differences in network needs and costs of using cloud services as opposed to deploying at the edge...
In Edge AI we can get low latency and faster inference due to processing taking place locally on the network. Upcoming 5G will be of great use to provide low latency also. Without the use of Edge AI we have to bear the cost of cloud services as the device may lack capacity to process the outputs in real time frame provided.


## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1) This can be used in a surveillance scene where it counts the people :
    For eg in a home where 5 rich people live and scared of theives it can count if more than 5 people are there and can raise the alarm as programed.
2) It can be used in schools :
    It can be used to keep track of the attendace by adding a slight feature which can identify each student. It will just then count the students and log them into the database.


## Assess Effects on End User Needs

Lighting :  The more the lightning the better will be the accuracy of the detection.

Model accuracy :  Model accuracy plays a major role. The better model accuracy we acheive the better results we get.

Camera focal length : In simple words high focal length gives focused images with which we can achieve good accuracy of the model. 

Image size :   Image size directly effects the processing time. The higher the accuracy the more time it will take to process the image frame and better will be the accuracy.


