
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream> 
#include <vector>

using namespace std;

char* readTextFromFile(string pathToFile);
int calculateHashCPU(char* text, int patternLen);
int moduloCPU(int x, int N);
__device__ int calculateHash(char* text, int patternLen);
__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
__device__ bool compareText(size_t length, char* text, char* pattern);
__device__ int modulo(int x, int N);
__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen);

int main()
{
    int numberOfCores = 3;
    char* text = readTextFromFile("test.txt");
    char* d_text;
    char* pattern = "abb";
    int numberOfChars = strlen(text);
    int patternLength = strlen(pattern);

    cout << endl << "Tekst:" << endl;
    cout << text << endl;
    cout << "__________" << endl;

    cout << "Wzorzec:" << endl;
    cout << pattern << endl;
    cout << "__________" << endl;
       

    int hashOfPattern = calculateHashCPU(pattern, patternLength);

    cout << hashOfPattern << endl;

    int pieceLen = numberOfChars / patternLength;

    if (numberOfChars % patternLength > 0) {
        pieceLen++;
    }

    cudaMalloc((char**)&d_text, numberOfChars * sizeof(char));
    cudaMemcpy(d_text, text, numberOfChars, cudaMemcpyHostToDevice);

    dim3 block(numberOfCores);   

    rabinKarp<<<1, numberOfCores>>>(text, numberOfChars, pattern, patternLength, hashOfPattern, pieceLen);

    cudaFree(d_text);

    return 0;
}

__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen) {       
    if (threadIdx.x == 1) {
        pieceLen -= (patternLength - 1);
    }

    char* pieceOfText = &text[(threadIdx.x - 1) * pieceLen];    

    int hashOfPieceOfText = calculateHash(pieceOfText, patternLength);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            printf("%d ", (threadIdx.x - 1) * pieceLen);
        }
    }

    printf("%d ", hashOfPieceOfText);

    for (int i = 1; i <= sizeof(pieceOfText) / sizeof(char); i++) {
        hashOfPieceOfText = moveHash(pieceOfText[i - 1], pieceOfText[i + patternLength - 1], hashOfPieceOfText, patternLength);

        pieceOfText = &pieceOfText[i];

        if (hashOfPattern == hashOfPieceOfText) {
            if (compareText(patternLength, pieceOfText, pattern)) {
                printf("%d ", (threadIdx.x - 1) * pieceLen + i);
            }
        }
    }
}

__device__ bool compareText(size_t length, char* text, char* pattern) {
    for (int i = 0; i < length; i++) {
        if (text[i] != pattern[i]) {
            return false;
        }
    }

    return true;
}

__device__ int calculateHash(char* text, int patternLen) {
    int mod = 23;
    int alphabetLen = 26;
    int result = 0;
    int exponent = static_cast<int>(sizeof(text) / sizeof(char)) - 1;

    for (int i = 0; i < patternLen; i++) {    
        if (text[i] != NULL && text[i] <= 'Z') {
            text[i] += 32;
        }

        result += (text[i] - 'a') * static_cast<int>(pow(alphabetLen, exponent));
        exponent--;
    }

    return modulo(result, mod);
}

__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen) {
    int mod = 23;
    int alphabetLen = 26;

    int multiplier = (int)pow(alphabetLen, textLen - 1);

    if (oldChar <= 'Z') {
        oldChar += 32;
    }

    int valueWithoutOldChar = (oldValue - (multiplier * (oldChar - 'a')));

    int valueWithNewChar = valueWithoutOldChar * alphabetLen;

    if (newChar <= 'Z') {
        newChar += 32;
    }

    valueWithNewChar += (newChar - 'a');

    return modulo(valueWithNewChar, mod);
}

__device__ int modulo(int x, int N) {
    return (x % N + N) % N;
}


int moduloCPU(int x, int N)
{
  return (x % N + N) % N;
}

int calculateHashCPU(char* text, int patternLen) {
    int mod = 23;
    int alphabetLen = 26;
    int result = 0;
    int exponent = static_cast<int>(sizeof(text) / sizeof(char)) - 1;

    for (int i = 0; i < patternLen; i++) {
        if (text[i] <= 'Z') {
            text[i] += 32;
        }

        result += (text[i] - 'a') * static_cast<int>(pow(alphabetLen, exponent));
        exponent--;
    }

    return moduloCPU(result, mod);
}

char* readTextFromFile(string pathToFile) {
    ifstream inFile;
    stringstream strStream;

    inFile.open(pathToFile);

    strStream << inFile.rdbuf();
    string resultString = strStream.str();
    char* result = new char[resultString.length()];
    strcpy(result, resultString.c_str());

    return result;
}
