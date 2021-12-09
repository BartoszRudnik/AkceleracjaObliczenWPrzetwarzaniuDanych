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
void showMenu(string& fileNameText, string& patternText);
void showDataInfo(long numberOfChars, int patternLength, string pattern, int blockSize, int gridSize);
void showTimeInfo(long long seconds, long long milisec, long long microsec);
int calculateHashCPU(char text[], int patternLen);
int moduloCPU(int x, int N);
int powWithModulo(float x, int exp, int mod);
__device__ int calculateHash(char text[], int patternLen);
__device__ int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
__device__ bool compareText(size_t length, char* text, char* pattern);
__device__ int modulo(int x, int N);
__device__ int powWithModulo(float x, int exp, int mod);
__global__ void rabinKarp(char* text, int numberOfChars, char* pattern, int patternLength, int hashOfPattern, int threadPart, int textLen);

int main()
{
    
    string fileName;
    string strPattern;
    char* pattern = new char[100];
    char* d_text;
    char* d_pattern;

    showMenu(fileName, strPattern);

    char* text = readTextFromFile(fileName);

    strcpy(pattern, strPattern.c_str());
    int patternLength = strlen(pattern);
    long numberOfChars = strlen(text);
    int hashOfPattern = calculateHashCPU(pattern, patternLength);
    int multiplier = 1;
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rabinKarp, 0, 0);
    gridSize = minGridSize;

    while (minGridSize * blockSize < numberOfChars / (patternLength * multiplier) + 1) {
        multiplier++;
    }

    int pieceLen = patternLength * multiplier;

    showDataInfo(numberOfChars, patternLength, pattern, blockSize, gridSize);

    cudaMalloc((char**)&d_pattern, patternLength * sizeof(char));
    cudaMalloc((char**)&d_text, numberOfChars * sizeof(char));

    auto start = chrono::system_clock::now();

    cudaMemcpy(d_pattern, pattern, patternLength, cudaMemcpyHostToDevice);


    cudaStream_t stream;;

    cudaStreamCreate(&stream);
    

    cudaMemcpyAsync(d_text, text, numberOfChars, cudaMemcpyHostToDevice, stream);
    rabinKarp << <gridSize, blockSize, 0, stream >> > (d_text, numberOfChars, d_pattern, patternLength, hashOfPattern, pieceLen, multiplier);
    

    cudaStreamDestroy(stream);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cout << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    auto end = chrono::system_clock::now();
    auto elapsed = end - start;
    auto microsec = chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    auto milisec = chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto seconds = chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    showTimeInfo(seconds, milisec, microsec);

    cudaFree(d_pattern);
    cudaFree(d_text);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    return 0;
}


__global__ void rabinKarp(char* text, int numberOfChars, char* pattern, int patternLength, int hashOfPattern, int pieceLen, int multiplier) {

    long id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id > 0) {
        id = id * multiplier * patternLength - 1;
        pieceLen += patternLength - 1;
    }

    if (id <= numberOfChars - patternLength) {

        int numberOfIterations = pieceLen - patternLength;

        char* pieceOfText = &text[id];

        int hashOfPieceOfText = calculateHash(pieceOfText, patternLength);

        if (hashOfPattern == hashOfPieceOfText) {
            if (compareText(patternLength, pieceOfText, pattern)) {
                printf("Znaleziono od indeksu: %d \n", id);
            }
        }

        for (int i = 1; i <= numberOfIterations; i++) {
            int oldHash = hashOfPieceOfText;
            hashOfPieceOfText = moveHash(pieceOfText[i - 1], pieceOfText[i + patternLength - 1], hashOfPieceOfText, patternLength);

            if (hashOfPattern == hashOfPieceOfText) {
                if (compareText(patternLength, pieceOfText + i, pattern)) {
                    printf("Znaleziono od indeksu: %d \n", id + i);
                }
            }
        }
    }
    else {
        return;
    }
}

__device__ void rabinKarpPart() {

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
            text[i] += 32;
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
        oldChar += 32;
    }
    if (newChar != NULL && newChar <= 'Z' && newChar >= 'A') {
        newChar += 32;
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
    //return x % N;
    return ((x % N) + N) % N;
}

__device__ int powWithModulo(float x, int exp, int mod) {
    if (exp == 0) {
        return 1;
    }

    float base = x;
    for (int i = 1; i < exp; i++) {
        base *= x;
        base = (int)fmodf(base, mod);
    }
    return (int)base;
}

int powWithModuloCPU(float x, int exp, int mod) {
    if (exp == 0) {
        return 1;
    }
    float base = x;
    for (int i = 1; i < exp; i++) {
        base *= x;
        base = (int)fmodf(base, mod);
    }
    return (int)base;
}



int moduloCPU(int x, int N)
{
    //return x % N;
    return ((x % N) + N) % N;
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

void showMenu(string& fileNameText, string& patternText) {
    cout << "KARP-RABIN GPU" << endl << endl;

    while (fileNameText.size() < 1) {
        cout << "Type file name: ";
        getline(cin, fileNameText);
    }

    while (patternText.size() < 1) {
        cout << "Type pattern: ";
        getline(cin, patternText);
    }
}

void showDataInfo(long numberOfChars, int patternLength, string pattern, int blockSize, int gridSize) {
    cout << "---------------" << endl;
    cout << "Text length: " << numberOfChars << endl;
    cout << "---------------" << endl;

    cout << "Pattern: " << "\"" << pattern << "\"" << endl;
    cout << "Pattern length: " << patternLength << endl;
    cout << "---------------" << endl;

    cout << "GPU param:" << endl;
    cout << "Block size: " << blockSize << " Grid size: " << gridSize << endl;
    cout << "---------------" << endl;
}

void showTimeInfo(long long seconds, long long milisec, long long microsec) {
    cout << "---------------" << endl;
    cout << "Time: ";
    if (seconds > 0) {
        cout << seconds << "." << milisec % 1000 << " sec " << endl;
    }
    else if (milisec > 0) {
        cout << milisec << "." << microsec % 1000 << " milisec " << endl;
    }
    else {
        cout << microsec << " microsec " << endl;
    }
}