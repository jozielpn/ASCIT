/* Classificador */

package classifier;

import java.util.ArrayList;
import java.util.List;
import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodRBF;

public class Classifier {
  private static SOM neural_network; //Rede neural SOM.
  private static int neural_network_dimension; //Parâmetro dimensão da rede neural.
  private static int neural_network_width; //Largura da rede neural.
  private static int neural_network_height; //Altura da rede neural.
  private static int neural_network_inputs; //Número de entradas da rede neural, que corresponde a dimensão do vetor descritor F.
  private static int neural_network_outputs; //Número de saídas da rede neural, que corresponde ao seu número total de neurônios.
  
  private static BasicTrainSOM competitive_training; //Treinamento competitivo.
  private static NeighborhoodRBF neighborhood_function; //Função de vizinhança.
  private static final double LEARNING_RATE = 0.01; //Taxa de aprendizado. Especifica quanto aplicar por iteração.
  private static final MLDataSet training_set = null; //Criação de um conjunto de treinamento vazio.
  private static int iterations; //Quantidade de iterações que será utilizada no treinamento.
  private static final double INITIAL_LEARNING_RATE = 1.0; //Taxa de aprendizado inicial.
  private static final double FINAL_LEARNING_RATE = 0.005; //Taxa de aprendizado final.
  public static int initial_neighborhood_radius = 50; //Raio de vizinhança inicial.
  private static final int FINAL_NEIGHBORHOOD_RADIUS = 1; //Raio de vizinhança final.
  
  //Parâmetro tipo de seleção (Seção 5.2.2). Opções:
  //  0 -> Seleção randômica
  //  1 -> Seleção randômica por grupo
  private static final int SELECTION_TYPE = 1;
  
  public static List<List<MLData>> t_set; //Conjunto T.
  private static Integer n_group; //Variável para selecionar um dos N grupos do conjunto T.
  
  //Variável para selecionar um vetor descritor F de um dos grupos do conjunto T.
  private static Integer vector_f;
  
  public static List<List<MLData>> k_set; //Conjunto K.
  private static List<List<Integer>> vector_w_set;
  private static List<Integer> vector_w; //Vetor W.
  private static int vector_w_element;
  private static int vector_w_sum;
  private static int tm; //Valor tm: Quantidade de padrões do menor grupo do conjunto K.
  private static int neural_network_output; //Saída da rede neural.
  private static double z_partial_error; //Erro parcial Z.
  private static double neural_network_error; //Erro da rede neural.
  private static int continue_verification;
  private static boolean is_ordered; //Variável de controle do algoritmo Bubble Sort.
  
  private static int vector_w1_size; //Número de elementos do primeiro vetor W.
  private static int vector_w2_size; //Número de elementos do segundo vetor W.
  private static int vector_w1_average; //Média aritmética dos elementos do primeiro vetor W.
  private static int vector_w2_average; //Média aritmética dos elementos do segundo vetor W.
  private static int vector_w1_d1; //Valor da diferença D1 do primeiro vetor W.
  private static int vector_w1_d2; //Valor da diferença D2 do primeiro vetor W.
  private static int vector_w2_d1; //Valor da diferença D1 do segundo vetor W.
  private static int vector_w2_d2; //Valor da diferença D2 do segundo vetor W.
  private static int greatest_difference_value; //Valor da maior diferença.
  private static int greatest_difference_selector; //Seletor da maior diferença.
  
  //Função para implementar a etapa Inicialização.
  public static void initializeNeuralNetwork(int vector_f_dimension) {
    neural_network_dimension = 5 * t_set.size();
    neural_network_width = neural_network_dimension;
    neural_network_height = neural_network_dimension;
    neural_network_outputs = neural_network_width * neural_network_height;
    neural_network_inputs = vector_f_dimension;
    neural_network = new SOM(neural_network_inputs, neural_network_outputs);//Criação da rede neural SOM.
    
    //Atribuição de valores aleatórios dentro do intervalo [-1,1] para o vetor de pesos de cada neurônio da rede neural.
    neural_network.reset();
  }
  
  //Função para implementar a etapa Treinamento.
  public static void trainNeuralNetwork() {
    iterations = 40 * t_set.size(); //Cálculo do número de iterações do treinamento.
    
    //Criação da função de vizinhança do tipo Gaussiana 2D.
    neighborhood_function = new NeighborhoodRBF(RBFEnum.Gaussian, neural_network_width, neural_network_height);
    
    //Criação do treinamento competitivo.
    competitive_training = new BasicTrainSOM(neural_network, LEARNING_RATE, training_set, neighborhood_function);
    
    //Configuração do treinamento competitivo para não forçar um neurônio vencedor.
    competitive_training.setForceWinner(false);
    
    //Configuração do decremento automático:
    //Ele diminuirá automaticamente o raio de vizinhança e a taxa de aprendizado dos valores iniciais aos valores finais.
    competitive_training.setAutoDecay(
      iterations,
      INITIAL_LEARNING_RATE,
      FINAL_LEARNING_RATE,
      initial_neighborhood_radius,
      FINAL_NEIGHBORHOOD_RADIUS
    );
    
    //Execução do treinamento da rede neural de acordo com o número de iterações.
    n_group = t_set.size() - 1;
    for (int i = 0; i < iterations; i++) {
      if (SELECTION_TYPE == 0)
        n_group = (int) (Math.random() * t_set.size());
      else {
        n_group++;
        if(n_group == t_set.size()) n_group = 0;
      }
      vector_f = (int) (Math.random() * t_set.get(n_group).size());
      
      //Aplicação de um vetor descritor F como entrada da rede neural para o seu treinamento.
      competitive_training.trainPattern(t_set.get(n_group).get(vector_f));
      
      competitive_training.autoDecay();
    }
  }
  
  //Função para implementar a etapa Avaliação.
  public static double evaluateNeuralNetwork() {
    //Fase Experimentação.
    //Geração um vetor W para cada textura da imagem I e cálculo do valor tm (Seção 5.2.4).
    vector_w_set = new ArrayList();
    tm = k_set.get(0).size();
    for (int i = 0; i < k_set.size(); i++) {
      vector_w = new ArrayList();
      
      //Busca pelo número de padrões do menor grupo do conjunto K (Valor tm).
      if (tm > k_set.get(i).size()) tm = k_set.get(i).size();
      
      for (int j = 0; j < k_set.get(i).size(); j++) {
        //Aplicação de todos os vetores de um grupo N do conjunto K na rede neural.
        neural_network_output = neural_network.classify(k_set.get(i).get(j));
        
        //Todas as saídas geradas pela rede neural nesse processo são usadas para formar o vetor W.
        vector_w.add(neural_network_output);
      }
      do { //O vetor W tem seus elementos ordenados de forma crescente através do algoritmo Bubble Sort.
        is_ordered = true;
        for(int j = 0; j < (vector_w.size() - 1) ; j++)
          if(vector_w.get(j) > vector_w.get(j + 1)) {
            vector_w_element = vector_w.get(j);
            vector_w.set(j, vector_w.get(j + 1));
            vector_w.set((j + 1), vector_w_element);
            is_ordered = false;
          }
      }while (!is_ordered);
      vector_w_set.add(vector_w);
    }
    
    z_partial_error = 100.0 / (tm - 3); //Cálculo do erro parcial Z.
    z_partial_error += 0.001; //Adição de pequeno incremento para possibilitar que: z_partial_error * (tm - 3) >= 100.0

    //Fase Verificação.
    neural_network_error = 0.0;
    while(neural_network_error < 100.0) {
      updateNeuralNetworkError();
      if(continue_verification == 0) break;
    }
    if (neural_network_error > 100.0) neural_network_error = 100.0;
    return neural_network_error;
  }
  
  private static void updateNeuralNetworkError() {
    continue_verification = 0;
    sortVectorWSet();

    //Busca por possível intersecção.
    for (int i = 0; i < (vector_w_set.size() - 1) && continue_verification == 0; i++) {
      //Cálculo do tamanho de cada vetor W envolvido no seguinte teste de intersecção.
      vector_w1_size = vector_w_set.get(i).size();
      vector_w2_size = vector_w_set.get(i + 1).size();

      if (vector_w_set.get(i).get(vector_w1_size - 1) >= vector_w_set.get(i + 1).get(0)) { //Houve uma intersecção.
        
        //A fase Verificação poderá ser executada novamente caso o novo erro da rede neural ainda seja menor que 100.
        continue_verification = 1;

        //Cálculo da média de cada vetor W envolvido na intersecção.
        vector_w1_average = getVectorWAverage(vector_w_set.get(i));
        vector_w2_average = getVectorWAverage(vector_w_set.get(i + 1));
        
        //Cálculo das diferenças.
        vector_w1_d1 = vector_w1_average - vector_w_set.get(i).get(0);
        vector_w1_d2 = vector_w_set.get(i).get(vector_w1_size - 1) - vector_w1_average;
        vector_w2_d1 = vector_w2_average - vector_w_set.get(i + 1).get(0);
        vector_w2_d2 = vector_w_set.get(i + 1).get(vector_w2_size - 1) - vector_w2_average;
        
        //Busca pela maior diferença.
        greatest_difference_selector = 0;
        greatest_difference_value = vector_w1_d1;
        if (vector_w1_d2 > greatest_difference_value) {
            greatest_difference_selector = 1;
            greatest_difference_value = vector_w1_d2;
        }
        if (vector_w2_d1 > greatest_difference_value) {
            greatest_difference_selector = 2;
            greatest_difference_value = vector_w2_d1;
        }
        if (vector_w2_d2 > greatest_difference_value) greatest_difference_selector = 3;
        
        switch (greatest_difference_selector) {
            case 0: {
                //A maior diferença é a diferença D1 do primeiro vetor W.
                //O elemento da primeira posição do vetor W usado no cálculo desta diferença D1 é removido.
                vector_w = vector_w_set.get(i);
                vector_w.remove(0);
                vector_w_set.set(i, vector_w);
                break;
            }
            case 1: {
                //A maior diferença é a diferença D2 do primeiro vetor W.
                //O elemento da última posição do vetor W usado no cálculo desta diferença D2 é removido.
                vector_w = vector_w_set.get(i);
                vector_w.remove(vector_w1_size - 1);
                vector_w_set.set(i, vector_w);
                break;
            }
            case 2: {
                //A maior diferença é a diferença D1 do segundo vetor W.
                //O elemento da primeira posição do vetor W usado no cálculo desta diferença D1 é removido.
                vector_w = vector_w_set.get(i + 1);
                vector_w.remove(0);
                vector_w_set.set(i + 1, vector_w);
                break;
            }
            case 3: {
                //A maior diferença é a diferença D2 do segundo vetor W.
                //O elemento da última posição do vetor W usado no cálculo desta diferença D2 é removido.
                vector_w = vector_w_set.get(i + 1);
                vector_w.remove(vector_w2_size - 1);
                vector_w_set.set(i + 1, vector_w);
                break;
            }
            default: { break; }
        }
        neural_network_error += z_partial_error;
      }
    }
  }
  
  //O conjunto de vetores W tem seus vetores ordenados de forma crescente de acordo com suas médias aritméticas.
  private static void sortVectorWSet() {
    do { //Uso do algoritmo Bublesort para fazer a ordenação.
      is_ordered = true;
      for(int i = 0; i < (vector_w_set.size() - 1) ; i++) {
        if(getVectorWAverage(vector_w_set.get(i)) > getVectorWAverage(vector_w_set.get(i + 1))) {
          vector_w = vector_w_set.get(i);
          vector_w_set.set(i, vector_w_set.get(i + 1));
          vector_w_set.set((i + 1), vector_w);
          is_ordered = false;
	}
      }
    } while (!is_ordered);
  }
  
  //Calcula a média aritmética dos valores de um vetor W.
  private static int getVectorWAverage(List<Integer> vector_w) {
    vector_w_sum = 0;
    for (int i = 0; i < vector_w.size(); i++)
        vector_w_sum += vector_w.get(i);
    return vector_w_sum / vector_w.size();
  }
}