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
int powWithModulo(float x, int exp, int mod);
__device__ int calculateHash(char text[], int patternLen);
__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
__device__ bool compareText(size_t length, char* text, char* pattern);
__device__ int modulo(int x, int N);
__device__ int powWithModulo(float x, int exp, int mod);
__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen);

int main()
{

    char* text = readTextFromFile("test.txt");
    char* d_text;
    char* d_pattern;
    string strPattern;
    char pattern[] = "feel will oh it we";
    int numberOfChars = strlen(text);
    int patternLength = strlen(pattern);

    cout << "Type pattern: ";
    getline(cin, strPattern);
    strcpy(pattern, strPattern.c_str());

    cout << endl << "________" << endl;
    cout << "Tekst:" << endl;

    cout << endl << "Text length: " << numberOfChars << endl;
    cout << "________" << endl;

    cout << "Wzorzec: " << "\"" << pattern << "\"" << endl;
    cout << endl << "Pattern length: " << patternLength << endl;
    cout << "________" << endl;

    int hashOfPattern = calculateHashCPU(pattern, patternLength);

    int combinations = numberOfChars - patternLength + 1;
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rabinKarp, 0, combinations);

    gridSize = (numberOfChars - patternLength + blockSize) / blockSize;

    cout << "GPU param:" << endl;
    cout << endl << "Block size: " << blockSize << " Gird size: " << gridSize << endl;
    cout << "________" << endl;

    int pieceLen = patternLength;

    auto start = chrono::system_clock::now();

    cudaMalloc((char**)&d_pattern, patternLength * sizeof(char));
    cudaMalloc((char**)&d_text, numberOfChars * sizeof(char));
    cudaMemcpy(d_pattern, pattern, patternLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_text, text, numberOfChars, cudaMemcpyHostToDevice);

    rabinKarp << <gridSize, blockSize >> > (d_text, numberOfChars, d_pattern, patternLength, hashOfPattern, pieceLen);
    cudaDeviceSynchronize();

    auto end = chrono::system_clock::now();
    auto elapsed = end - start;
    auto microsec = chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    auto milisec = chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto seconds = chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    cout << endl << "Time: ";
    if (seconds > 0) {
        cout << seconds << "." << milisec % 1000 << " sec " << endl;
    }
    else if (milisec > 0) {
        cout << milisec << "." << microsec % 1000 << " milisec " << endl;
    }
    else {
        cout << microsec << " microsec " << endl;
    }

    cudaFree(d_pattern);
    cudaFree(d_text);

    return 0;
}


__global__ void rabinKarp(char* text, int textLength, char* pattern, int patternLength, int hashOfPattern, int pieceLen) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id <= textLength - patternLength) {

        int numberOfIterations = 0;
        if (pieceLen > patternLength) {
            numberOfIterations = pieceLen - patternLength;
        }
        char* pieceOfText = &text[id];

        int hashOfPieceOfText = calculateHash(pieceOfText, patternLength);

        if (hashOfPattern == hashOfPieceOfText) {
            if (compareText(patternLength, pieceOfText, pattern)) {
                printf("Znaleziono od indeksu: %d \n", id);
            }
        }


        for (int i = 1; i <= numberOfIterations; i++) {
            hashOfPieceOfText = moveHash(pieceOfText[i - 1], pieceOfText[i + patternLength - 1], hashOfPieceOfText, patternLength);

            if (hashOfPattern == hashOfPieceOfText) {
                if (compareText(patternLength, pieceOfText + i, pattern)) {
                    printf("Znaleziono od indeksu: %d \n", id + i);
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

        result += text[i] * powWithModulo(alphabetLen, exponent, mod);
        exponent--;
    }

    return modulo(result, mod);
}

__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen) {
    int mod = 101;
    int alphabetLen = 256;

    int multiplier = powWithModulo(alphabetLen, textLen - 1, mod);

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
    return x % N;
}

__device__ int powWithModulo(float x, int exp, int mod) {
    float base = x;
    for (int i = 1; i < exp; i++) {
        base *= base;
        base = (int)fmodf(x, mod);
    }
    return (int)base;
}

int powWithModuloCPU(float x, int exp, int mod) {
    float base = x;
    for (int i = 1; i < exp; i++) {
        base *= base;
        base = (int)fmodf(x, mod);
    }
    return (int)base;
}



int moduloCPU(int x, int N)
{
    return x % N;
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

        result += text[i] * powWithModuloCPU(alphabetLen, exponent, mod);
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