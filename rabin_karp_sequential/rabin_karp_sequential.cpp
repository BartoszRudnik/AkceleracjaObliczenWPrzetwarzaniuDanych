#include <fstream>
#include <iostream>
#include <sstream> 
#include<math.h>
#include <cctype>
#include <chrono>

using namespace std;

int alphabetLen = 256;
int mod = 101;

void showDataInfo(int numberOfChars, int patternLength, string pattern);
void showTimeInfo(long long seconds, long long milisec, long long microsec);
void showMenu(string& fileNameText, string& patternText);
string readTextFromFile(string pathToFile);
int calculateHash(string text);
int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
bool compareText(size_t length, string text, string pattern);
int modulo(double x, int N);
int modulo(int x, int N);

int main()
{
    string pattern;
    string fileName;

    showMenu(fileName, pattern);

    string text = readTextFromFile(fileName);

    showDataInfo(text.size(), pattern.size(), pattern);

    auto start = chrono::system_clock::now();

    int hashOfPattern = calculateHash(pattern);
    size_t patternLength = pattern.length();
    long textLength = text.size();
    string pieceOfText = text.substr(0, patternLength);
    int hashOfPieceOfText = calculateHash(pieceOfText);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {

            cout << "Znaleziono od indeksu: 0" << endl;
        }
    }
    
    for (int i = 1; i <= textLength - patternLength; i++) {
        hashOfPieceOfText = moveHash(text[i - 1], text[i + patternLength - 1], hashOfPieceOfText, patternLength);
        

        if (hashOfPattern == hashOfPieceOfText) {
            pieceOfText = text.substr(i, patternLength);
            if (compareText(patternLength, pieceOfText, pattern)) {
                cout << "Znaleziono od indeksu: " + to_string(i) << endl;
            }
        }
    }    

    auto end = chrono::system_clock::now();
    auto elapsed = end - start;
    auto microsec = chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    auto milisec = chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto seconds = chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    showTimeInfo(seconds, milisec, microsec);

    return 0;
}
void showMenu(string& fileNameText, string& patternText) {
    cout << "KARP-RABIN CPU SEQUENTIAL" << endl << endl;

    while (fileNameText.size() < 1) {
        cout << "Type file name: ";
        getline(cin, fileNameText);
    }

    while (patternText.size() < 1) {
        cout << "Type pattern: ";
        getline(cin, patternText);
    }
}

void showDataInfo(int numberOfChars, int patternLength, string pattern) {
    cout << "---------------" << endl;
    cout << "Text length: " << numberOfChars << endl;
    cout << "---------------" << endl;

    cout << "Pattern: " << "\"" << pattern << "\"" << endl;
    cout << "Pattern length: " << patternLength << endl;
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

bool compareText(size_t length, string text, string pattern) {
    for (int i = 0; i < length; i++) {
        if (tolower(text[i]) != tolower(pattern[i])) {
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

int modulo(double x, int N) {
    return (int)fmod((fmod(x, N) + N), N);
}

int modulo(int x, int N) {
    return ((x % N) + N) % N;
}

int moveHash(char oldChar, char newChar, int oldValue, size_t textLen) {
    int multiplier = modulo(pow(alphabetLen, textLen - 1), mod);

    int valueWithoutOldChar = modulo(oldValue - (multiplier * (tolower(oldChar))), mod);

    int valueWithNewChar = modulo(valueWithoutOldChar * alphabetLen, mod);
    valueWithNewChar += (tolower(newChar));

    return modulo(valueWithNewChar, mod);
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
