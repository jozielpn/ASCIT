/* Algoritmo dos Testes do Capítulo 6 */
package classifier;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class Test {
  //Variável para armazenar a localização do projeto ASCIT.
  private static final String ASCIT_PATH = "C:/Users/JozielPN/Projects/";
  
  //Variável para possibilitar a seleção do teste a ser executado. Opções:
  //  1 -> Executa o Teste 1
  //  2 -> Executa o Teste 2
  //  3 -> Executa o Teste 3
  private static final int TEST = 1;
  
  //Variável para armazenar o nome das texturas da imagem I usada em cada teste.
  private static final String[][] textures = {
  /* Teste 1 */{ "Superior", "Inferior" },
  /* Teste 2 */{ "First", "Second", "Third" },
  /* Teste 3 */{ "First Superior", "Second Superior", "First Inferior", "Second Inferior" }
  };
  
  //Variável para possibilitar a seleção do algoritmo a ser utilizado. Opções:
  //  0 -> Algoritmo SIFT
  //  1 -> Algoritmo SURF
  private static final int ALGORITHM_OPTION = 0;
  
  //Variável para armazenar o nome do algoritmo selecionado.
  private static String algorithm_name;
  
  //Variável para armazenar o nome completo e sem abreviações do algoritmo selecionado.
  private static String algorithm_fullname;
  
  //Variável para armazenar a dimensão do vetor descritor F gerado pela aplicação Extrator de características,
  //de acordo com o algoritmo selecionado.
  private static int vector_f_dimension;
  
  private static String vector_f_file_path;
  private static File vector_f_file;
  private static MLData vector_f;
  private static List<MLData> n_group;
  
  private static BufferedReader buffered_reader;
  private static String file_line;
  private static int vector_f_element_selector;
  private static char[] vector_f_element_char;
  private static String vector_f_element_string;
  private static float vector_f_element;
  
  //Lista para armazenar todos os erros gerados por todas as redes neurais instanciadas durante a execução do ciclo O (Seção 6.2).
  private static List<Double> error_list;
  
  private static int a;
  private static float e;
  private static int total_hit;
  private static float general_error;
  
  //Lista para armazenar o resultado gerado em cada iteração do ciclo U:
  //Cada resultado é formado pelos valores dos fatores acerto total e erro geral.
  private static List<String> result_list;
  
  private static File result_file;
  private static BufferedWriter buffered_writer;
  
  public static void main(String[] args) { //Função principal para a execução do algoritmo.
    System.out.print("Classifier\n\n");
    if (ALGORITHM_OPTION == 0) {
      vector_f_dimension = 128;
      algorithm_name = "SIFT";
      algorithm_fullname = "(Scale Invariant Features Transform)";
    } else {
      vector_f_dimension = 64;
      algorithm_name = "SURF";
      algorithm_fullname = "(Speeded-Up Robust Features)";
    }
    System.out.print("Algorithm: " + algorithm_name + " " + algorithm_fullname + "\n\n");
    System.out.print("Executing Test " + TEST + ". Please wait. Processing...\n\n");
    runTest();
    System.out.print("\n\nTest " + TEST + " executed successfully!\n");
  }
  
  private static void runTest() { //Função para executar o teste selecionado.
    buildSet("Training", 5); //Construção do conjunto T (Seção 5.2.2).
    buildSet("Evaluation", 10); //Construção do conjunto K (Seção 5.2.4).
    
    //Execução do ciclo U descrito na Seção 6.3.
    result_list = new ArrayList();
    Classifier.initial_neighborhood_radius = 2;
    System.out.print("U Cycle Iterations: |");
    for(int u = 0; u < 98; u++, Classifier.initial_neighborhood_radius++) {
      System.out.print((u + 1) + "|");
      total_hit = 0;
      general_error = 0;
      
      //Execução do ciclo P do método de teste descrito na Seção 6.2 para calcular os valores A e E.
      for(int p = 0; p < 10; p++) {
        error_list = new ArrayList();
        
        //Execução do ciclo O do método de teste descrito na Seção 6.2.
        for(int o = 0; o < 50; o++) {
          Classifier.initializeNeuralNetwork(vector_f_dimension); //Inicialização da rede neural.
          Classifier.trainNeuralNetwork(); //Treinamento da rede neural.
          error_list.add(Classifier.evaluateNeuralNetwork()); //Cálculo do erro da rede neural.
        }
        
        //Cálculo do valores A e E.
        a = 0;
        e = 0;
        for (Double error : error_list) {
          if (error == 0.0) a++;
          e += error;
        }
        e = e / 50;
        
        general_error += e;
        total_hit += a;
      }
      //É feito o cálculo dos fatores acerto total e erro geral para cada valor do raio inicial de vizinhança.
      general_error = general_error / 10;
      total_hit = total_hit / 10;
      
      result_list.add("Initial Neighborhood Radius: " + String.valueOf(Classifier.initial_neighborhood_radius) + "\t\tTotal Hit: " +
        String.valueOf(total_hit) + "\t\tGeneral Error: " + String.format("%.2f", general_error));
    }
    generateResult();
  }
  
  //Função para construir o conjunto T ou o conjunto K de acordo com os parâmetros disponibilizados.
  //O conjunto T é utilizado na etapa Treinamento descrita na seção 5.2.2.
  //O conjunto K é utilizado na etapa Avaliação descrita na seção 5.2.4.
  private static void buildSet(String test_step, int n_group_size) {
    if(n_group_size < 10) Classifier.t_set = new ArrayList();
    else Classifier.k_set = new ArrayList();
    for (int i = 0; i < textures[TEST-1].length; i++) {
      n_group = new ArrayList();
      for (int j = 0; j < n_group_size ; j++) {
        vector_f_file_path = ASCIT_PATH + "ASCIT/Vector F/Test " + String.valueOf(TEST) + "/" + test_step + "/" + 
          algorithm_name + "/" + textures[TEST-1][i] + " Texture " + String.valueOf(j + 1) + ".txt";
        vector_f_file = new File(vector_f_file_path);
        importVectorF();
        n_group.add(vector_f);
      }
      if(n_group_size < 10) Classifier.t_set.add(n_group);
      else Classifier.k_set.add(n_group);
    }
  }
  
  private static void importVectorF() {
    vector_f = new BasicMLData(vector_f_dimension);
    vector_f_element_char = new char[10];
    try (FileReader file_reader = new FileReader(vector_f_file)) {
      buffered_reader = new BufferedReader(file_reader);
      file_line = buffered_reader.readLine(); //Lê uma linha do arquivo.
      vector_f_element_selector = 0;
      for(int i = 0, j = 0; i < file_line.length(); i++, j++)
        if(file_line.charAt(i) == '|' || i == file_line.length() - 1) { //Fim de um elemento do vetor descritor F.
          if (i == file_line.length() -1) vector_f_element_char[j] = file_line.charAt(i);
          vector_f_element_string = new String(vector_f_element_char);
          vector_f_element = Float.parseFloat(vector_f_element_string);
          vector_f.setData(vector_f_element_selector, vector_f_element);
          vector_f_element_selector++;
          vector_f_element_char = new char[10];
          j = 0;
        } else vector_f_element_char[j] = file_line.charAt(i);
      try { buffered_reader.close(); }
      catch (IOException exception) { System.out.println("Error closing the BufferedReader: " + exception.getMessage()); }
    }
    catch (IOException exception) { System.err.println("\nError opening file: " + exception.getMessage()); }
  }
  
  private static void generateResult() {
    result_file = new File(ASCIT_PATH + "ASCIT/Results/Test " + TEST + " - " + algorithm_name + ".txt");
    if (!result_file.exists())
      try { result_file.createNewFile(); }
      catch (IOException exception) { System.out.println("Error creating file: " + exception.getMessage()); }
    try (FileWriter file_writer = new FileWriter(result_file)) {
      buffered_writer = new BufferedWriter(file_writer);
      for (int i = 0; i < result_list.size(); i++) {
        if (i > 0) buffered_writer.newLine();
        buffered_writer.write(result_list.get(i));
      }
      try { buffered_writer.close(); }
      catch (IOException exception) { System.out.println("Error closing the BufferedWriter: " + exception.getMessage()); }
    }
    catch (IOException exception) { System.err.println("\nError opening file: " + exception.getMessage()); }
  }
}