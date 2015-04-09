/*
 * (sockets)
 *   C++ sockets on Unix and Windows
 *   Copyright (C) 2002
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *   See <http://cs.baylor.edu/~donahoo/practical/CSockets/practical/>
 *
 * (opencv)
 *  Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 *  Released to public domain under terms of the BSD Simplified license.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <map>

#include "PracticalSocket.h"

#include <cstdlib>
#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp" 

using namespace cv;
using namespace std;

const int WIDTH = 100;
const int HEIGHT = 100;

const unsigned int RCVBUFSIZE = 1024;    // Size of receive buffer
TCPSocket* handleTCPClient(TCPSocket *sock); // TCP client handling function
void sendToTCPSocket(TCPSocket *sock, std::string); // TCP client send

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::cout << "Filename: " << filename << std::endl;
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int count = 0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat im = imread(path, 0);
            labels.push_back(atoi(classlabel.c_str()));
	    cv::resize(im, im, cv::Size(WIDTH, HEIGHT));
	    images.push_back(im);
        }
        count++;
    }
    std::cout << "Lines:  " << count << std::endl;
    std::cout << "Images: " << images.size() << std::endl;
}

static map<int, std::string> labelToName(std::string filename, char separator = ';') {
  map<int, std::string> mymap;
  std::ifstream file(filename.c_str(), ifstream::in);
  string line, label, name;
  while(getline(file, line)) {
    stringstream liness(line);
    getline(liness, label, separator);
    getline(liness, name);
    if(!label.empty() && !name.empty()) {
      mymap[atoi(label.c_str())] = name;
//      cout << label << " = " << name << endl;
    }
  }
  return mymap;
}


int main(int argc, const char *argv[]) {
    
    unsigned short echoServPort = 6789;

    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <names_file> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 4) {
        output_folder = string(argv[3]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    map<int, std::string> mymap = labelToName(string(argv[2]));

    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    
    // Read in the data. This can fail if no valid input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can doTCPSocket*
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:

    string path = "";
    Mat src; 

    string classLabel;
    int testLabel;

    // START SERVER //
    try 
    {
        TCPServerSocket servSock(echoServPort);     // Server Socket object
        TCPSocket *clientSocket;

        for (;;) {   // Run forever

            clientSocket = handleTCPClient(servSock.accept());       // Wait for a client to connect
	    src = imread("temp.jpeg", 0);
	
	    // Resize any received images.
	    cv::resize(src, src, cv::Size(WIDTH, HEIGHT));
 
	    // Predict the results.
	    double confidence;
            int predictedLabel;
            model->predict(src, predictedLabel, confidence);
            string result_message = format("Predicted class = %d. Confidence = %f\n", predictedLabel, confidence);
            string result_name = mymap[predictedLabel] + "\n";
            cout << result_message << endl;

            clientSocket->send(result_name.c_str(), result_message.size());
            cout << "name sent : " << result_name << endl;

//   	    sendToTCPSocket(clientSocket, result_message);
	    delete clientSocket;

        }
    } 
    catch (SocketException &e) {
        cerr << "SocketException: " << e.what() << endl;
        exit(1);
    }

    return 0;
}


// TCP client handling function
TCPSocket* handleTCPClient(TCPSocket *sock) {
  cout << "Handling client ";
  try {
    cout << sock->getForeignAddress() << ":";
  } catch (SocketException e) {
    cerr << "Unable to get foreign address" << endl;
  }
  try {
    cout << sock->getForeignPort();
  } catch (SocketException e) {
    cerr << "Unable to get foreign port" << endl;
  }
  cout << endl;

  std::ofstream outfile;
  outfile.open("temp.jpeg", ios::out | ios::binary);
  outfile.clear();

  // Send received string and receive again until the end of transmission
  char echoBuffer[RCVBUFSIZE];
  int recvMsgSize;
  cout << "About to read...\n";
  std::string message = "hello client\n";
  sock->send(message.c_str(), message.size());

  sock->recv(echoBuffer, sizeof (int) );
  int result = 0;
  int bytesRead = 0;

  for (unsigned n = 0; n < sizeof( result ); n++)
      result = (result << 8) +echoBuffer[ n ];

  cout << "Photo size: " << result << endl;

//  while ( ( recvMsgSize = sock->recv(echoBuffer, RCVBUFSIZE)) > 0) { // Zero means end

  while (bytesRead < result) {
    recvMsgSize = sock->recv(echoBuffer, RCVBUFSIZE);
    bytesRead += recvMsgSize;
//    recvMsgSize = sock->recv(echoBuffer, RCVBUFSIZE);
    cout << "recvMsgSize : " << recvMsgSize << endl;
    // Echo message back to client
//    cout << "."; 
//    sock->send(echoBuffer, recvMsgSize);
//    const char *temp = echoBuffer;
    outfile.write(echoBuffer, recvMsgSize);
  };

  outfile.flush();
  outfile.close();

//  message = "goodbye client\n";
//  sock->send(message.c_str(), message.size());
  cout << "Finished reading...\n";
  return sock;
}


void sendToTCPSocket(TCPSocket *sock, std::string message) {
  cout << "Sending result to client ";
  try {
    cout << sock->getForeignAddress() << ":";
  } catch (SocketException e) {
    cerr << "Unable to get foreign address" << endl;
  }
  try {
    cout << sock->getForeignPort();
  } catch (SocketException e) {
    cerr << "Unable to get foreign port" << endl;
  }
  cout << endl;

  cout << "Sending message : " << message << endl;
  sock->send(message.c_str(), message.size());

  delete sock;
}

