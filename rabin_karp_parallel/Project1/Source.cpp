#include <fstream>
#include <iostream>
#include <sstream> 
#include<math.h>
#include <cctype>
#include<vector>
#include<thread>
#include <chrono>

using namespace std;

int alphabetLen = 256;
const int numberOfThreads = 4;
int mod = 101;

string readTextFromFile(string pathToFile);
int calculateHash(string text);
int moveHash(char oldChar, char newChar, int oldValue, int textLen);
bool compareText(size_t length, string text, string pattern);
int modulo(double x, int N);
int modulo(int x, int N);
vector<string> divideText(string text, int patternLength, int numberOfThreads, vector<int>* indexing);
void rabinKarp(string text, string pattern, int startingIndex);

int main(){
    
    if (thread::hardware_concurrency() != 0) {
        numberOfThreads = thread::hardware_concurrency();
    }

    string text = readTextFromFile("test.txt");
    string pattern;

    cout << "Podaj wzorzec: ";
    getline(cin, pattern);

    cout << endl << "Tekst:" << endl;
    cout << text << endl;
    cout << "__________" << endl;

    cout << "Wzorzec:" << endl;
    cout << pattern << endl;
    cout << "__________" << endl;
    
    vector<int> indexing;
    vector<string> dividedText = divideText(text, pattern.length(), numberOfThreads, &indexing);
    vector<thread> threads(dividedText.size());
    
    for (int i = 0; i < dividedText.size(); i++) {
        cout << dividedText[i] << endl;
    }
    
    cout << "__________" << endl;

    auto start = chrono::system_clock::now();
    
    for (int i = 0; i < dividedText.size(); i++) {
        threads[i] = thread(rabinKarp, dividedText[i], pattern, indexing[i]);
    }

    for (int i = 0; i < dividedText.size(); i++) {
        threads[i].join();
    }
    
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

    return 0;
}

void rabinKarp(string text, string pattern, int startingIndex) {
    int hashOfPattern = calculateHash(pattern);
    int patternLength = pattern.length();
    int textLength = text.length();
    string pieceOfText = text.substr(0, patternLength);
    int hashOfPieceOfText = calculateHash(pieceOfText);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            cout << "Znaleziono od indeksu: " << startingIndex << endl;
        }
    }

    for (int i = 1; i <= textLength - patternLength; i++) {
        hashOfPieceOfText = moveHash(text[i - 1], text[i + patternLength - 1], hashOfPieceOfText, patternLength);
        pieceOfText = text.substr(i, patternLength);

        if (hashOfPattern == hashOfPieceOfText) {
            if (compareText(patternLength, pieceOfText, pattern)) {
                cout << "Znaleziono od indeksu: " + to_string(i + startingIndex) << endl;
            }
        }
    }
}

vector<string> divideText(string text, int patternLength, int numberOfThreads, vector<int>* indexing) {
    vector<string> result;

    size_t baseLength = text.length();
    int combinations = baseLength - patternLength + 1;
    int combinationsPerThread = combinations / numberOfThreads;

    int shift = 0;
    int start = 0;
    for (int i = 0; i < numberOfThreads; i++) {

        if (i == numberOfThreads - 1) {
            start = i * (combinationsPerThread);
            shift = text.length() - start;
            result.push_back(text.substr(start, shift));
            indexing->push_back(start);
        }
        else {
            start = i * (combinationsPerThread);
            shift = (i + 1) * (combinationsPerThread)+(patternLength - 1) - start;
            result.push_back(text.substr(start, shift));
            indexing->push_back(start);
        }
    }

    return result;
}

bool compareText(int length, string text, string pattern) {
    for (int i = 0; i < length; i++) {
        if (text[i] != pattern[i]) {
            return false;
        }
    }

    return true;
}

int calculateHash(string text) {
    int result = 0;
    int exponent = text.length() - 1;

    for (int i = 0; i < text.length(); i++) {
        result += (tolower(text[i])) * modulo(pow(alphabetLen, exponent), mod);
        exponent--;
    }

    return modulo(result, mod);
}

int moveHash(char oldChar, char newChar, int oldValue, int textLen) {
    int multiplier = modulo(pow(alphabetLen, textLen - 1), mod);
    
    int valueWithoutOldChar = modulo(oldValue - (multiplier * (tolower(oldChar))), mod);

    int valueWithNewChar = modulo(valueWithoutOldChar * alphabetLen, mod);
    valueWithNewChar += (tolower(newChar));

    return modulo(valueWithNewChar, mod);
}

int modulo(double x, int N) {
    return (int)fmod((fmod(x, N) + N), N);
}

int modulo(int x, int N) {
    return (x % N + N) % N;
}



string readTextFromFile(string pathToFile) {
    ifstream inFile;
    stringstream strStream;
    string text;

    inFile.open(pathToFile);

    strStream << inFile.rdbuf();
    
    for (string line; getline(strStream, line); ) {
        text += line + " ";
    }

    return text;
}
