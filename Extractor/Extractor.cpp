/* Extrator de características */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int algorithm_option = 0;
string algorithm_name;
string algorithm_fullname;
Ptr<Feature2D> algorithm;

int vector_f_dimension;
vector<float> vector_f;
char vector_f_element[10];

string ascit_path;
string input_image_path;
string vector_f_file_path;

Mat input_image;
Mat vector_set_c;
vector<KeyPoint> keypoints;

int64 runtime_start_ticks = 0;
int64 runtime_ticks = 0;
double runtime = 0.0;

int task_option;
int task_errors;
vector<string> tasks{ "Single Extraction", "Test 1", "Test 2", "Test 3" };


int test_index;
bool features_extraction_success;

/*
Variável para selecionar cada grupo de sub-imagens utilizados nos testes descritos no Capítulo 6:
  Training    ->  É gerado um vetor descritor F para cada sub-imagem do passo 2 descrito no método de teste citado na Seção 6.3.
  Evaluation  ->  É gerado um vetor descritor F para cada sub-imagem do passo 3 descrito no método de teste citado na Seção 6.3.
*/
vector<string> test_step{ "Training", "Evaluation" };

vector<vector<string>> textures{ //Variável para armazenar o nome das texturas da imagem I usada em cada teste do capítulo 6.
  /* Teste 1 */{ "Superior", "Inferior" },
  /* Teste 2 */{ "First", "Second", "Third" },
  /* Teste 3 */{ "First Superior", "Second Superior", "First Inferior", "Second Inferior" }
};

string end_application;

//Declaração de funções.
void initializeAlgorithm();
void extractFeatures();
void generateVectorF();
void storeVectorF();
void printVectorFDetails();
void singleExtraction();
void test();

//Função principal de execução da aplicação.
int main() {
  cout << "Features Extractor (OpenCV " << CV_VERSION << ")";

  //Variável para a seleção do algoritmo a ser utilizado. Opções:
  //  0 -> Algoritmo SIFT
  //  1 -> Algoritmo SURF
  algorithm_option = 0;

  initializeAlgorithm();

  cout << "\n\n\nSelected Algorithm: " << algorithm_name << " (" << algorithm_fullname << ")";

  //Variável para armazenar a localização do projeto ASCIT.
  ascit_path = "C:/Users/JozielPN/Projects/";

  //Atribuição do caminho da imagem de entrada.
  //Essa atribuição será usada apenas para a execução da tarefa Extração Única.
  input_image_path = ascit_path + "ASCIT/Images/" + tasks[0] + "/Flowers.jpg";

  //Atribuição do caminho de criação do arquivo de texto contendo o vetor descritor F.
  //Essa atribuição será usada apenas para a execução da tarefa Extração Única.
  vector_f_file_path = ascit_path + "ASCIT/Vector F/" + tasks[0] + "/Vector F (" + algorithm_name + ").txt";

  //Variável para selecionar a tarefa a ser executada usando o algoritmo selecionado. Opções:
  //  0 -> Extração Única   -> Gerar um único vetor descritor F para uma única imagem de entrada.
  //  1 -> Teste 1          -> Gerar um vetor descritor F para cada sub-imagem utilizada no Teste 1 descrito na seção 6.4.
  //  2 -> Teste 2          -> Gerar um vetor descritor F para cada sub-imagem utilizada no Teste 2 descrito na seção 6.5.
  //  3 -> Teste 3          -> Gerar um vetor descritor F para cada sub-imagem utilizada no Teste 3 descrito na seção 6.6.
  task_option = 0;

  //Execução da tarefa selecionada.
  if (task_option > -1 && task_option < 4) {
    cout << "\n\n\nExecuting " << tasks[task_option] << "...\n\n\n";
    task_errors = 0; //Contador de erros.
    if (task_option > 0) test();
    else singleExtraction();
  }
  else {
    cout << "\n\n\nYou selected an invalid task option.";
    task_errors = -1;
  }

  //Exibição de mensagem para indicar o fim da execução da aplicação e os possíveis erros ocorridos durante a execução da tarefa selecionada.
  if (task_errors > 0) {
    cout << "\n" << tasks[task_option] << " could not be completed successfully.";
    if (task_errors > 1) cout << "\n\n\nThere were " << task_errors << " errors!";
    else cout << "\n\n\nAn error has occurred!";
  }
  else if (task_errors == 0) cout << "\n" << tasks[task_option] << " completed successfully!";

  cout << "\n\n\nEnter anything to close the application: ";
  cin >> end_application;
  return 1;
}

//Início da 1° Etapa: Descrição de pontos de interesse (Seção 5.1.1) -> Inicialização do algoritmo selecionado.
void initializeAlgorithm() {
  if (algorithm_option == 0) { //Caso o algoritmo SIFT tenha sido selecionado.
    algorithm_name = "SIFT"; //Variável para armazenar o nome do algoritmo.
    algorithm_fullname = "Scale Invariant Features Transform"; //Variável para armazenar o nome completo do algoritmo sem abreviações.
    vector_f_dimension = 128;

    //O algoritmo SIFT é instanciado e inicializado com os seguintes parâmetros configurados:
    //(int nfeatures = all, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6)
    algorithm = SIFT::create();
  }
  else { //Caso algoritmo SURF tenha sido selecionado.
    algorithm_name = "SURF"; //Variável para armazenar o nome do algoritmo.
    algorithm_fullname = "Speeded-Up Robust Features"; //Variável para armazenar o nome completo do algoritmo sem abreviações.
    vector_f_dimension = 64;

    //O algoritmo SURF é instanciado e inicializado com os seguintes parâmetros configurados:
    //(double hessianThreshold = 800, int nOctaves = 4, int nOctaveLayers = 3, bool extended = false, bool upright = false)
    algorithm = SURF::create(800);
  }
  vector_f.resize(vector_f_dimension, 0.0);
}

//Fim da 1° Etapa: Descrição de pontos de interesse (Seção 5.1.1) -> Geração do conjunto C de vetores descritores.
void extractFeatures() {
  imread(input_image_path, IMREAD_GRAYSCALE).copyTo(input_image); //Leitura da imagem de entrada.

  if (input_image.empty()) {
    cout << "ERROR!" << "\nCould not load the Input Image:\n" << input_image_path << "\n\n";
    features_extraction_success = false;
  }
  else {
    /*
    Para calcular o tempo decorrido durante o processo de extração de características da imagem de entrada,
    é utilizada a função getTickCount(), que retorna o atual número de pulsos do relógio de máquina após um determinado evento,
    como por exemplo, quando a máquina foi ligada.
    */

    runtime_start_ticks = getTickCount();

    //Extração de características: detecção dos pontos de interesse e computação dos vetores descritores.
    algorithm->detectAndCompute(input_image, Mat(), keypoints, vector_set_c);

    runtime_ticks = getTickCount() - runtime_start_ticks;

    //Calcula o tempo total decorrido em milisegundos, através da frequencia do relógio de máquina.
    runtime = runtime_ticks / ((double)getTickFrequency()) * 1000.0;

    features_extraction_success = true;
  }
}

//2° Etapa: Geração de vetor descritor (Seção 5.1.2) -> Cálculo do vetor descritor F, o qual representa a imagem de entrada.
void generateVectorF() {
  for (unsigned int i = 0; i < keypoints.size(); i++)
    for (int j = 0; j < vector_f_dimension; j++)
      vector_f[j] += vector_set_c.at<float>(i, j);

  for (int j = 0; j < vector_f_dimension; j++)
    vector_f[j] /= keypoints.size();

  //Se o algoritmo SIFT foi selecionado, é feita a normalização do vetor descritor F através da divisão de todos os seus valores por 100.
  if (algorithm_option == 0)
    for (int j = 0; j < 128; j++)
      vector_f[j] /= 100;
}

//Função para armazenar o vetor descritor F em um arquivo de texto.
void storeVectorF() {
  ofstream file(vector_f_file_path); //Cria ou abre arquivo.
  for (int i = 0; i < vector_f_dimension; i++) {
    if (i > 0) file << "|";
    snprintf(vector_f_element, 10, "%f", vector_f[i]);
    file << vector_f_element;
  }
  file.close(); //Fecha arquivo.
}

//Função para exibir detalhes relacionados à geração do vetor F.
void printVectorFDetails() {
  cout << "Input Image:\t\t\t" << input_image_path;
  cout << "\nVector F File:\t\t\t" << vector_f_file_path;

  //Exibição da quantidade de pontos de interesse que foram detectados na imagem de entrada.
  cout << "\nKeypoints Quantity:\t\t" << keypoints.size();

  //Exibição do tempo de execução do processo de detecção e descrição dos pontos de interesse.
  cout << "\nFeatures Extraction Runtime:\t" << runtime << " milliseconds\n\n";
}

void singleExtraction() {
  extractFeatures();
  if (features_extraction_success) {
    generateVectorF();
    storeVectorF();
    printVectorFDetails();
  }
  else task_errors++;
}

void test() {
  test_index = task_option - 1;
  for (int s = 0; s < 2; s++) {
    for (unsigned int i = 0; i < textures[test_index].size(); i++)
      for (int j = 1; j <= (5 * (s + 1)); j++) {
        input_image_path = ascit_path + "ASCIT/Images/" + tasks[task_option] + "/" + test_step[s] + "/" +
          textures[test_index][i] + " Texture " + to_string(j) + ".png";
        extractFeatures();
        if (features_extraction_success) {
          generateVectorF();
          vector_f_file_path = ascit_path + "ASCIT/Vector F/" + tasks[task_option] + "/" + test_step[s] + "/" +
            algorithm_name + "/" + textures[test_index][i] + " Texture " + to_string(j) + ".txt";
          storeVectorF();
          printVectorFDetails();
        }
        else task_errors++;
      }
  }
}