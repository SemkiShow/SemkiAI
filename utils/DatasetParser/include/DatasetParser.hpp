#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <sys/stat.h>

// void ParseHTML(char* path, char* outputDirectory);
void ParseCSV(std::string path, std::vector<std::vector<std::string>>* dataset);
