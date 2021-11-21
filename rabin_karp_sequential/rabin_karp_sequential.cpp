#include <fstream>
#include <iostream>
#include <sstream> 
#include<math.h>
#include <cctype>
#include <chrono>

using namespace std;

int alphabetLen = 26;
int mod = 23;

string readTextFromFile(string pathToFile);
int calculateHash(string text);
int moveHash(char oldChar, char newChar, int oldValue, size_t textLen);
bool compareText(size_t length, string text, string pattern);
int modulo(int x, int N);
int modulo(double x, int N);

int main()
{
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
    
    auto start = chrono::system_clock::now();

    int hashOfPattern = calculateHash(pattern);
    size_t patternLength = pattern.length();
    size_t textLength = text.length();
    string pieceOfText = text.substr(0, patternLength);
    int hashOfPieceOfText = calculateHash(pieceOfText);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            cout << "Znaleziono od indeksu: 0" << endl;
        }
    }

    for (int i = 1; i <= textLength - patternLength; i++) {
        hashOfPieceOfText = moveHash(text[i - 1], text[i + patternLength - 1], hashOfPieceOfText, patternLength);
        pieceOfText = text.substr(i, patternLength);

        if (hashOfPattern == hashOfPieceOfText) {            
            if (compareText(patternLength, pieceOfText, pattern)) {
                cout << "Znaleziono od indeksu: " + to_string(i) << endl;
            }
        }       
    }
    
    auto end = chrono::system_clock::now();
    auto elapsed = end - start;
    
    cout << endl << "__________" << endl;
    cout << "Czas wykonania algorytmu: " << endl << 
        chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " milisec" << endl << 
        chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microsec" << endl;

    return 0;
}

bool compareText(size_t length, string text, string pattern) {
    for (int i = 0; i < length; i++) {
        if (text[i] != pattern[i]) {
            return false;
        }
    }
    
    return true;
}

int calculateHash(string text) {
    int result = 0;
    int exponent = static_cast<int>(text.length()) - 1;

    for (int i = 0; i < text.length(); i++) {
        result += (tolower(text[i])) * modulo(pow(alphabetLen, exponent), mod);
        exponent--;
    }

    return modulo(result, mod);
}

int modulo(int x, int N) {
    return (x % N + N) % N;
}

int modulo(double x, int N) {
    return (int)fmod((fmod(x, N) + N), N);
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
