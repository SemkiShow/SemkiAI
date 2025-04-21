#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <vector>

std::vector<std::string> Split(std::string input, char delimiter = ' ');
void ParseHTML(char* path, char* outputDirectory);
std::vector<std::vector<std::string>> ParseCSV(std::string path);
