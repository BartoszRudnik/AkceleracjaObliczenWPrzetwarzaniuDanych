
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream> 
#include <vector>
#include <chrono>

using namespace std;

char* readTextFromFile(string pathToFile);
int calculateHashCPU(char text[], int patternLen);
int moduloCPU(int x, int N);
__device__ int calculateHash(char text[], int patternLen);
__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
__device__ bool compareText(size_t length, char* text, char* pattern);
__device__ int modulo(int x, int N);
__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen);

int main()
{   
    int device_count = 0;
    int multiprocesors = 0;
    int blocksPerMultiproc = 0;
    int threadsPerBloc = 0;

    cudaGetDeviceCount(&device_count);
    cudaDeviceGetAttribute(&multiprocesors, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&blocksPerMultiproc, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&threadsPerBloc, cudaDevAttrMaxThreadsPerBlock, 0);

    cout << "devices: " << device_count << " multiprocesors: " << multiprocesors <<
        " blocks per multiproc: " << blocksPerMultiproc <<
        " threads per block: " << threadsPerBloc << endl;

    char* text = readTextFromFile("test.txt");
    char* d_text;
    char* d_pattern;
    char pattern[] = "joy";
    int numberOfChars = strlen(text);
    int patternLength = strlen(pattern);

    cout << endl << "Tekst:" << endl;
    cout << text << endl;
    cout << "__________" << endl;

    cout << "Wzorzec:" << endl;
    cout << pattern << endl;
    cout << "__________" << endl;

    int hashOfPattern = calculateHashCPU(pattern, patternLength);

    cout << "Pattern hash: " << hashOfPattern << endl << "__________" << endl;

    int pieceLen = patternLength;
    int numberOfCores = 1;
    int numberOfBlocks = 96;
    int neededCores = numberOfChars - patternLength + 1;

    if (neededCores > numberOfBlocks) {
        numberOfCores = neededCores / numberOfBlocks;

        if (neededCores % numberOfBlocks > 0) {
            numberOfCores++;
        }
    }
    else if (neededCores < numberOfBlocks) {
        numberOfBlocks = neededCores;
    }

    cudaMalloc((char**)&d_pattern, patternLength * sizeof(char));
    cudaMalloc((char**)&d_text, numberOfChars * sizeof(char));
    cudaMemcpy(d_pattern, pattern, patternLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, text, numberOfChars, cudaMemcpyHostToDevice);

    dim3 block(numberOfCores);   

    cout << numberOfChars << " " << neededCores << " " << numberOfBlocks << " " << numberOfCores << endl;

    auto start = chrono::system_clock::now();

    rabinKarp<<<numberOfBlocks, numberOfCores>>>(d_text, numberOfChars, d_pattern, patternLength, hashOfPattern, pieceLen);
    cudaDeviceSynchronize();

    auto end = chrono::system_clock::now();
    auto elapsed = end - start;
    auto microsec = chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    auto milisec = chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto seconds = chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    cout << "Time: ";
    if (seconds > 0) {
        cout << seconds << " sec " << endl;
    }
    if (milisec > 0) {
        cout << milisec << " milisec " << endl;
    }

    cudaFree(d_pattern);
    cudaFree(d_text);

    return 0;
}

__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen) {       
    int id = (blockIdx.x * threadIdx.x) + (blockIdx.x * blockDim.x) + threadIdx.x;

    char* pieceOfText = &text[id * pieceLen];    

    if (id > 0) {
        pieceOfText -= patternLength - 1;
    }

    int hashOfPieceOfText = calculateHash(pieceOfText, patternLength);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            if (id == 0) {
                printf("Znaleziono od indeksu: %d \n", id * pieceLen);
            }
            else {
                printf("Znaleziono od indeksu: %d \n", id * pieceLen - (patternLength - 1));
            }            
        }
    }

    int numberOfIterations = pieceLen;
    if (id > 0) {
        numberOfIterations += patternLength - 1;
    }

    for (int i = 1; i <= numberOfIterations - patternLength; i++) {
        hashOfPieceOfText = moveHash(pieceOfText[i - 1], pieceOfText[i + patternLength - 1], hashOfPieceOfText, patternLength);
        
        if (hashOfPattern == hashOfPieceOfText) {           
            if (compareText(patternLength, pieceOfText + i, pattern)) {
                if (id == 0) {
                    printf("Znaleziono od indeksu: %d \n", id * pieceLen + i);
                }
                else {
                    printf("Znaleziono od indeksu: %d \n", id * pieceLen - (patternLength - 1) + i);
                }               
            }
        }
    }
}

__device__ bool compareText(size_t length, char* text, char* pattern) {    
    for (int i = 0; i < length; i++) {
        if (text[i] != NULL && text[i] <= 'Z' && text[i] >= 'A') {
            text[i] += ('a' - 'A');
        }

        if (text[i] != pattern[i]) {
            return false;
        }
    }

    return true;
}

__device__ int calculateHash(char text[], int patternLen) {
    int mod = 101;
    int alphabetLen = 256;
    int result = 0;
    int exponent = patternLen - 1;

    for (int i = 0; i < patternLen; i++) {
        if (text[i] != NULL && text[i] <= 'Z' && text[i] >= 'A') {
            text[i] += ('a' - 'A');
        }

        result += text[i] * modulo(pow(alphabetLen, exponent), mod);
        exponent--;
    }

    return modulo(result, mod);
}

__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen) {
    int mod = 101;
    int alphabetLen = 256;

    int multiplier = modulo(pow(alphabetLen, textLen - 1), mod);

    if (oldChar != NULL && oldChar <= 'Z' && oldChar >= 'A') {
        oldChar += ('a' - 'A');
    }
    int valueWithoutOldChar = modulo(oldValue - (multiplier * (oldChar)), mod);

    int valueWithNewChar = modulo(valueWithoutOldChar * alphabetLen, mod);


    if (newChar != NULL && newChar <= 'Z' && newChar >= 'A') {
        newChar += ('a' - 'A');
    }
    valueWithNewChar += newChar;

    return modulo(valueWithNewChar, mod);
}

__device__ int modulo(int x, int N) {
    return (x % N + N) % N;
}

int moduloCPU(double x, int N) {
    return (int)fmod((fmod(x, N) + N), N);
}

int moduloCPU(int x, int N)
{
  return (x % N + N) % N;
}

int calculateHashCPU(char text[], int patternLen) {
    int mod = 101;
    int alphabetLen = 256;
    int result = 0;
    int exponent = patternLen - 1;

    for (int i = 0; i < patternLen; i++) {
        if (text[i] != NULL && text[i] <= 'Z' && text[i] >= 'A') {
            text[i] += ('a' - 'A');
        }

        result += text[i] * moduloCPU(pow(alphabetLen, exponent), mod);
        exponent--;
    }

    return moduloCPU(result, mod);
}

char* readTextFromFile(string pathToFile) {
    ifstream inFile;
    stringstream strStream;
    string text;

    inFile.open(pathToFile);

    strStream << inFile.rdbuf();    

    for (string line; getline(strStream, line); ) {
        text += line + " ";
    }

    char* result = new char[text.length()];
    strcpy(result, text.c_str());

    return result;
}
