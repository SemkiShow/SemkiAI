#include "DatasetParser.hpp"

std::vector<std::string> files;

int ListFiles(std::string path)
{
    struct stat sb;
 
    // Looping until all the items of the directory are
    // exhausted
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
 
        // Converting the path to const char * in the
        // subsequent lines
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        const char* outfilename_char = outfilename_str.c_str();
 
        // Testing whether the path points to a
        // non-directory or not If it does, displays path
        if (stat(outfilename_char, &sb) == 0 && !(sb.st_mode & S_IFDIR))
            files.push_back(outfilename_str);
        else
            ListFiles(outfilename_str);
    }
    return 0;
}

std::vector<std::string> Split(std::string input, char delimiter)
{
    std::vector<std::string> output;
    output.push_back("");
    int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] == delimiter)
        {
            index++;
            output.push_back("");
            continue;
        }
        output[index] += input[i];
    }
    return output;
}

// Work in progress
// void ParseHTML(char* path, char* outputDirectory)
// {
//     ListFiles(path);
//     std::cout << "There are a total of ";
//     std::cout << files.size();
//     std::cout << " files in the given path" << "\n";
//     std::cout << "Show all of them?(y/N) ";
//     std::string answer;
//     std::getline(std::cin, answer);
//     for (auto& tmp : answer) {
//         tmp = tolower(tmp);
//     }
//     if (/* answer.empty() ||  */answer == "y")
//     {
//         for (int i = 0; i < files.size(); i++)
//         {
//             std::cout << files[i] << "\n";
//         }
//     }

//     std::fstream input;
//     std::fstream output;
//     std::string file;
//     std::string fileContent;
//     char writePath[1024] = {0};
//     std::vector<std::string> tmp;
//     for (int i = 0; i < files.size(); i++)
//     {
//         input.open(files[i]);
//         while (std::getline(input, file))
//         {
//             // std::cout << file << "\n";
//             fileContent += file;
//             fileContent += "\n";
//         }
//         // std::cout << "------------------" << "\n";
//         // std::cout << fileContent << "\n";
//         input.close();

//         tmp = Split(files[i], '/');
//         strcat(writePath, outputDirectory);
//         strcat(writePath, "/");
//         // strcat(writePath, &tmp[tmp.size()]);
//         output.open(writePath);
//         std::cout << tmp[tmp.size()] << "\n";

//         file = "";
//         fileContent = "";
//         memset(writePath, 0, sizeof(writePath));
//         output.close();
//         tmp = std::vector<std::string>();
//     }
// }

void ParseCSV(std::string path, std::vector<std::vector<std::string>>* dataset)
{
    std::fstream datasetFile;
    datasetFile.open(path, std::ios::in);
    std::string buf = "";
    while (std::getline(datasetFile, buf))
        dataset->push_back(Split(buf, ','));
    datasetFile.close();
}
