#include <fstream>
#include <iostream>
#include <sstream> 
#include<math.h>
#include <cctype>
#include<vector>
#include<thread>

using namespace std;

int alphabetLen = 26;
const int numberOfThreads = 4;
int mod = 23;

string readTextFromFile(string pathToFile);
int calculateHash(string text);
int moveHash(char oldChar, char newChar, int oldValue, int textLen);
bool compareText(int length, string text, string pattern);
int modulo(int x, int N);
vector<string> divideText(string text, int patternLength, int numberOfThreads);
void rabinKarp(string text, string pattern, int startingIndex);

int main(){

    string text = readTextFromFile("test.txt");
    string pattern;

    cout << "Podaj wzorzec: ";
    cin >> pattern;

    cout << endl << "Tekst:" << endl;
    cout << text << endl;
    cout << "__________" << endl;

    cout << "Wzorzec:" << endl;
    cout << pattern << endl;
    cout << "__________" << endl;

    vector<string> dividedText = divideText(text, pattern.length(), 4);
    vector<thread> threads(dividedText.size());
    
    for (int i = 0; i < dividedText.size(); i++) {
        cout << dividedText[i] << endl;
    }

    for (int i = 0; i < dividedText.size(); i++) {
        int startingIndex = 0;

        if (i > 0) {
            int lengthOfPieceOfText = text.length() / numberOfThreads;

            startingIndex = i * lengthOfPieceOfText - (lengthOfPieceOfText - 1);
        }

        threads[i] = thread(rabinKarp, dividedText[i], pattern, startingIndex);
    }

    for (int i = 0; i < dividedText.size(); i++) {
        threads[i].join();
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

vector<string> divideText(string text, int patternLength, int numberOfThreads) {
    vector<string> result;

    int lengthOfPartOfText = text.length() / numberOfThreads;
    int num = text.length() % numberOfThreads;

    for (int i = 0; i <= text.length();) {
        if (i + lengthOfPartOfText < text.length()) {
            if (i > 0) {
                int patternLengthMinusOne = patternLength - 1;

                result.push_back(text.substr(i - patternLengthMinusOne, lengthOfPartOfText + patternLengthMinusOne));
            }
            else {
                result.push_back(text.substr(i, lengthOfPartOfText));
            }
        }
        else {
            if (i > 0) {
                int patternLengthMinusOne = patternLength - 1;

                result.push_back(text.substr(i - patternLengthMinusOne, num + patternLengthMinusOne));
            }
            else {
                result.push_back(text.substr(i, num));
            }
        }

        
        i += lengthOfPartOfText;
    }

    return result;
}

vector<string> divideText2(string text, int patternLength, int numberOfThreads) {
    vector<string> smallTexts;
    vector<string> result;

    for (int i = 0; i < patternLength; i++) {
        for (int j = i; j < text.length();) {
            if (j + patternLength <= text.length()) {
                smallTexts.push_back(text.substr(j, patternLength));
                j += patternLength;
            }
            else {
                break;
            }            
        }
    }

    int pieces = smallTexts.size() / numberOfThreads;
    int extra = smallTexts.size() % numberOfThreads;

    for (int i = 0; i < smallTexts.size(); i+= pieces) {
        string connected = "";

        for (int j = i; j < i + pieces; j++) {
            connected += smallTexts[j];
        }

        if (i + pieces >= smallTexts.size()) {
            for (int j = i + pieces; j < smallTexts.size(); j++) {
                connected += smallTexts[j];
            }
        }

        result.push_back(connected);
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
        result += (tolower(text[i]) - 'a') * pow(alphabetLen, exponent);
        exponent--;
    }

    return modulo(result, mod);
}

int modulo(int x, int N) {
    return (x % N + N) % N;
}

int moveHash(char oldChar, char newChar, int oldValue, int textLen) {
    int multiplier = (int)pow(alphabetLen, textLen - 1);

    int valueWithoutOldChar = (oldValue - (multiplier * (tolower(oldChar) - 'a')));

    int valueWithNewChar = valueWithoutOldChar * alphabetLen;
    valueWithNewChar += (tolower(newChar) - 'a');

    return modulo(valueWithNewChar, mod);
}

string readTextFromFile(string pathToFile) {
    ifstream inFile;
    stringstream strStream;

    inFile.open(pathToFile);

    strStream << inFile.rdbuf();

    return strStream.str();
}