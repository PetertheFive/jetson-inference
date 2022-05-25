/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoSource.h"
#include "videoOutput.h"

#include "poseNet.h"

#include <signal.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

const int endPos=820;
const int startPos=220;
const int scaleUp = 600;
const int cutLimit = 100 - (10000/scaleUp) ;

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: posenet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Run pose estimation DNN on a video/image stream.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", poseNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}

int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();

	/*
	 * load Apple logo
	 */
	cv::Mat logo,resized_logo;
	logo = cv::imread("images/apple.jpg", cv::IMREAD_COLOR);
	if(! logo.data ){// Check for invalid input
		LogError("detectnet:  failed to load images/apple.jpg\n");
        	return -1;
    	}
 
	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("posenet: failed to create input stream\n");
		return 0;
	}


	/*
	 * create output stream
	 */
/*	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("posenet: failed to create output stream\n");	
*/	

	/*
	 * create recognition network
	 */
	poseNet* net = poseNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("posenet: failed to initialize poseNet\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlayFlags = poseNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "links,keypoints"));
	
	
	//Rex:Show a initial screen
    	namedWindow( "Logo", WINDOW_NORMAL );
	setWindowProperty ("Logo", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    	imshow( "Logo", logo );
	waitKey(33);
    	imshow( "Logo", logo );
	waitKey(33);
    	imshow( "Logo", logo );
	waitKey(33);

	int resetCounter = 100;
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image image
		uchar3* image = NULL;

		if( !input->Capture(&image, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break;

			LogError("posenet: failed to capture next frame\n");
			continue;
		}

		// run pose estimation
		std::vector<poseNet::ObjectPose> poses;
		
		if( !net->Process(image, input->GetWidth(), input->GetHeight(), poses, overlayFlags) )
		{
			LogError("posenet: failed to process frame\n");
			continue;
		}
		
		LogInfo("posenet: detected %zu %s(s)\n", poses.size(), net->GetCategory());

		// render outputs
/*		if( output != NULL )
		{
			output->Render(image, input->GetWidth(), input->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}
*/

		if ( (poses.size()==0) && (resetCounter>0) ) {
			if(--resetCounter == 0) {
				LogVerbose("percent reset\n");
    				imshow( "Logo", logo );
				waitKey(33);
    				imshow( "Logo", logo );
				waitKey(33);
    				imshow( "Logo", logo );
				waitKey(33);
			}
		}	

		for(std::vector<poseNet::ObjectPose>::iterator pose=poses.begin(); pose!=poses.end(); ++pose)
		{
			int leftShoulder = pose->FindKeypoint(5);
			int rightShoulder = pose->FindKeypoint(6);
			/*int leftElbow = pose->FindKeypoint(7);
			int leftWrist = pose->FindKeypoint(9);
			int rightElbow = pose->FindKeypoint(8);
			int rightWrist = pose->FindKeypoint(10);
			int leftHip = pose->FindKeypoint(11);
			int rightHip = pose->FindKeypoint(12);
			int leftKnee = pose->FindKeypoint(13);
			int leftAnkle = pose->FindKeypoint(15);
			int rightKnee = pose->FindKeypoint(14);
			int rightAnkle = pose->FindKeypoint(16);*/

			if((leftShoulder!=-1) && (rightShoulder!=-1)) {
				resetCounter = 100;
				int middleOfShoulder = (pose->Keypoints[leftShoulder].x + pose->Keypoints[rightShoulder].x)/2;
				if(middleOfShoulder > endPos)
					middleOfShoulder = endPos;
				else if(middleOfShoulder < startPos)
					middleOfShoulder = startPos;
				LogVerbose("outform: shoulder = %f, %f, middleOfShoulder=%d", pose->Keypoints[leftShoulder].x, pose->Keypoints[rightShoulder].x, middleOfShoulder);
				int cutOff = (middleOfShoulder-startPos) * 100 / (endPos-startPos) * cutLimit/ 100;
                                //LogVerbose("cutOff = %d\n", cutOff);
                                if(cutOff<1) cutOff=1;
				else if(cutOff>99) cutOff=99;
				Mat croppedImage = logo(Rect( (1920-(1920*(100-cutOff)/100))/2, (1080-(1080*(100-cutOff)/100))/2, (1920*(100-cutOff)/100), (1080*(100-cutOff)/100) ));
				//LogVerbose("cutOff=%d, Rec(%d, %d)  (%d, %d) \n", cutOff, (1920-(1920*(100-cutOff)/100))/2, (1080-(1080*(100-cutOff)/100))/2, (1920*(100-cutOff)/100), (1080*(100-cutOff)/100));
				
	    			// Create a window for display.
                                imshow( "Logo", croppedImage );
                              // Show our image inside it.
                                waitKey(1);
				break;
			}
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("posenet: shutting down...\n");
	
	SAFE_DELETE(input);
//	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	LogVerbose("posenet: shutdown complete.\n");
	return 0;
}

