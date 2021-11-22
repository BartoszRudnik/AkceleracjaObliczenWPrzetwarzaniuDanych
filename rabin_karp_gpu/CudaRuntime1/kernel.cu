
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
    char* d_pattern;
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

    cout << "Pattern hash: " << hashOfPattern << endl << "__________" << endl;

    int pieceLen = numberOfChars / patternLength;

    if (numberOfChars % patternLength > 0) {
        pieceLen++;
    }

    cudaMalloc((char**)&d_pattern, patternLength * sizeof(char));
    cudaMalloc((char**)&d_text, numberOfChars * sizeof(char));
    cudaMemcpy(d_pattern, pattern, patternLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, text, numberOfChars, cudaMemcpyHostToDevice);

    dim3 block(numberOfCores);   

    rabinKarp<<<1, numberOfCores>>>(d_text, numberOfChars, d_pattern, patternLength, hashOfPattern, pieceLen);

    cudaFree(d_pattern);
    cudaFree(d_text);

    return 0;
}

__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen) {       
    char* pieceOfText = &text[threadIdx.x * pieceLen];    

    if (threadIdx.x > 0) {
        pieceOfText -= patternLength - 1;
    }

    int hashOfPieceOfText = calculateHash(pieceOfText, patternLength);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            if (threadIdx.x == 0) {
                printf("Znaleziono od indeksu: %d \n", threadIdx.x * pieceLen);
            }
            else {
                printf("Znaleziono od indeksu: %d \n", threadIdx.x * pieceLen - (patternLength - 1));
            }            
        }
    }

    int numberOfIterations = pieceLen;
    if (threadIdx.x > 0) {
        numberOfIterations += patternLength - 1;
    }

    for (int i = 1; i <= numberOfIterations - patternLength; i++) {
        hashOfPieceOfText = moveHash(pieceOfText[i - 1], pieceOfText[i + patternLength - 1], hashOfPieceOfText, patternLength);
        
        if (hashOfPattern == hashOfPieceOfText) {           
            if (compareText(patternLength, pieceOfText + i, pattern)) {
                if (threadIdx.x == 0) {
                    printf("Znaleziono od indeksu: %d \n", threadIdx.x * pieceLen + i);
                }
                else {
                    printf("Znaleziono od indeksu: %d \n", threadIdx.x * pieceLen - (patternLength - 1) + i);
                }               
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
    int exponent = patternLen - 1;

    for (int i = 0; i < patternLen; i++) {
        result += text[i] * modulo(pow(alphabetLen, exponent), mod);
        exponent--;
    }

    return modulo(result, mod);
}

__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen) {
    int mod = 23;
    int alphabetLen = 26;

    int multiplier = modulo(pow(alphabetLen, textLen - 1), mod);

    int valueWithoutOldChar = modulo(oldValue - (multiplier * (oldChar)), mod);

    int valueWithNewChar = modulo(valueWithoutOldChar * alphabetLen, mod);
    valueWithNewChar += newChar;

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
    int exponent = patternLen - 1;

    for (int i = 0; i < patternLen; i++) {
        if (text[i] != NULL && text[i] <= 'Z') {
            text[i] += 32;
        }

        result += text[i] * moduloCPU(pow(alphabetLen, exponent), mod);
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
