import argparse
import cv2
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
counter = 0
INPUT_STREAM = "Pedestrian_Detect_2_1_1.mp4"
INPUT_MODEL = "frozen_inference_graph.xml"
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU"):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        #if cpu_extension and "CPU" in device:
            #self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        print(type(self.input_blob))
        self.input_blob = "image_tensor"
        print(type(self.input_blob))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs['image_tensor'].shape
        #return self.network.inputs[self.input_blob].shape


    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    #required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-m", help=m_desc, default=INPUT_MODEL)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    counter=0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        obj = box[1] # OBJECT = 1 ie. person ( for coco dataset)
        #if conf >= 0.5:
        if conf >= 0.4 and obj == 1:
            counter += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    #cap = cv2.VideoCapture(0)
    cap.open(args.i)
    #cap.open(0)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    #out = cv2.VideoWriter('out_1.mp4', 0x7634706d, 30, (width,height))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('out_Pedestrian_Detect(0.3)_2_1_1.mp4', fourcc, 30, (width,height))
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
                # Write out the frame
            out.write(frame)

            cv2.imshow("DETECT{}".format(counter),frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
