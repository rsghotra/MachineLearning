using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DecisionTreeAlgo
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			DecisionTree dt = new DecisionTree();
			dt.ConstructTree();
			//dt.PrepareWekaData(@"/u3/rguttikonda/Desktop/weka/traindata.csv", dt.trainingDocs, dt.trainingSet);
			//dt.PrepareWekaData(@"/u3/rguttikonda/Desktop/weka/testdata.csv", dt.testDocs, dt.testSet);
		}
	}
	
	
	class DecisionTree
	{
		//Lists used to store the data from files
		public List<Document> trainingDocs = new List<Document>();
		public List<Document> testDocs = new List<Document>();
		public List<Label> trainingSet = new List<Label>();
		public List<Label> testSet = new List<Label>();
		List<Word> TotalWords = new List<Word>();
		
		public DecisionTree()
		{
			#region loading files
			LoadDataForDocs(@"/Users/rupinderghotra/Desktop/Files/trainData.txt", trainingDocs);
			trainingDocs.Add(new Document(){ documentId = 1017, words = new List<int>()});	//Adding the missing documentId
			Console.WriteLine("Training Docs Loaded: " + trainingDocs.Count);
			
			LoadDataForDocs(@"/Users/rupinderghotra/Desktop/Files/testData.txt", testDocs);
			Console.WriteLine("Test Docs Loaded: " + testDocs.Count);
			
			LoadDataForLabels(@"/Users/rupinderghotra/Desktop/Files/trainLabel.txt", trainingSet);	
			Console.WriteLine("Training Labels Loaded: " + trainingSet.Count);
			
			LoadDataForLabels(@"/Users/rupinderghotra/Desktop/Files/testLabel.txt", testSet);
			Console.WriteLine("Test Labels Loaded: " + testSet.Count);
			
			StreamReader wordsFile = new StreamReader(@"/Users/rupinderghotra/Desktop/Files/words.txt");
			int wordId = 1;
			string temp = wordsFile.ReadLine();
			while(temp != null)
			{
				TotalWords.Add(new Word() { wordId = wordId, word = temp});
				wordId++;
				
				temp = wordsFile.ReadLine();
			}
			
			Console.WriteLine("Total No. of Words: " + TotalWords.Count);
			#endregion
		}
		
		public void ConstructTree()
		{
			//Priority Queue used for construction of Tree -- contains leaves of the tree (ordered by information gain at each leaf)
			List<Node> priorityQueue = new List<Node>();
			
			List<Label> tempLabels = CloneLabelList(trainingSet);
			
			//creating root and adding to the tree
			Node root = createNodeWithMaxInfoGain(tempLabels);
			priorityQueue.Add(root);
			
			int NodeCount = 0;
			while(priorityQueue.Count > 0 && NodeCount<=100)
			{
				Node currentNode = priorityQueue[0];
				priorityQueue.Remove(currentNode);
				
				tempLabels = CloneLabelList(currentNode.trainingLabels);
				
				//create child for with word
				List<Label> tempLabelsWithWord = tempLabels.FindAll(p=> trainingDocs.Exists(d => d.documentId == p.documentId && d.words.Contains(currentNode.word.wordId)));
				currentNode.WithWord = createNodeWithMaxInfoGain(tempLabelsWithWord);
				
				//create child for without word
				List<Label> tempLabelsWithoutWord = tempLabels.FindAll(p=> trainingDocs.Exists(d => d.documentId == p.documentId && !d.words.Contains(currentNode.word.wordId)));
				currentNode.WithoutWord = createNodeWithMaxInfoGain(tempLabelsWithoutWord);
				
				currentNode.isLeafNode = false;
				currentNode.classLabel = -1;
				
				priorityQueue.Add(currentNode.WithWord);
				priorityQueue.Add(currentNode.WithoutWord);
				priorityQueue.Sort((x, y) => x.InformationGain.CompareTo(y.InformationGain));
				priorityQueue.Reverse();
				
				NodeCount++;
				
				Console.WriteLine("NodeCount:" + NodeCount + "TrainingAccuracy: " + TestAccuracyOfTree(root, trainingSet, trainingDocs));
				Console.WriteLine("NodeCount:" + NodeCount + "TestAccuracy: " + TestAccuracyOfTree(root, testSet, testDocs));
			}
		}
		
		#region Loading Documents functions
		//Loading documents
		void LoadDataForDocs(string filePath, List<Document> docs)
		{
			StreamReader docsFile = new StreamReader(filePath);
			string temp = docsFile.ReadLine();
			while(temp != null)
			{
				int docId = int.Parse(temp.Split('\t')[0]);
				int wordId = int.Parse(temp.Split('\t')[1]);
				List<int> twords = new List<int>();
				twords.Add(wordId);
				if(docs.Exists(p=>p.documentId == docId))
					docs.Find(p=>p.documentId == docId).words.Add(wordId);
				else
					docs.Add(new Document(){ documentId = docId, words = twords });
				
				temp = docsFile.ReadLine();
			}
		}
		
		//Loading labels
		void LoadDataForLabels(string filePath, List<Label> labels)
		{
			StreamReader labelsFile = new StreamReader(filePath);
			int docId = 1;
			string temp = labelsFile.ReadLine();
			while(temp != null)
			{
				int labelId = int.Parse(temp);
				labels.Add(new Label() { documentId = docId, label = labelId});
				docId++;
				
				temp = labelsFile.ReadLine();
			}
		}
		#endregion
		
		double InformationGain(Word word, List<Label> trainingLabels)
		{
			//Docs containing word
			List<Document> docsContainingWord = trainingDocs.FindAll(doc => doc.words.Contains(word.wordId));
			int l1withWord = trainingLabels.FindAll(p=>p.label == 1 && docsContainingWord.Exists(d=>d.documentId == p.documentId)).Count;
			int l2withWord = trainingLabels.FindAll(p=>p.label == 2 && docsContainingWord.Exists(d=>d.documentId == p.documentId)).Count;
			
			//Docs not containing word
			List<Document> docsNotContainingWord = trainingDocs.FindAll(doc => !doc.words.Contains(word.wordId));
			int l1withoutWord = trainingLabels.FindAll(p=>p.label == 1 && docsNotContainingWord.Exists(d=>d.documentId == p.documentId)).Count;
			int l2withoutWord = trainingLabels.FindAll(p=>p.label == 2 && docsNotContainingWord.Exists(d=>d.documentId == p.documentId)).Count;
			
			double entropyWithword = entropy(l1withWord, l2withWord);
			double entropyWithoutword = entropy(l1withoutWord, l2withoutWord);
			
			double total = (double)(l1withWord + l2withWord + l1withoutWord + l2withoutWord);
			double totalWithWord = (double)(l1withWord + l2withWord);
			double totalWithoutWord = (double)(l1withoutWord + l2withoutWord);
			
			double remainder = (totalWithWord/total) * entropyWithword + (totalWithoutWord/total) * entropyWithoutword;
			double ActualEntropyWithoutSplit = entropy(l1withWord + l1withoutWord, l2withWord+l2withoutWord);
			
			double IG = ActualEntropyWithoutSplit - remainder;
			
			return IG;
		}
		
		double entropy(int l1, int l2)
		{
			double x1 = l1/(double)(l1 + l2);
			double x2 = l2/(double)(l1 + l2);
			
			double val1=0.0, val2=0.0;
			if(x1 != 0)
				val1 = - x1 * Math.Log(x1) / Math.Log(2.0);
			if(x2 != 0)
				val2 = - x2 * Math.Log(x2) / Math.Log(2.0);
			
			return (val1+val2);
		}
		
		Node createNodeWithMaxInfoGain(List<Label> trainingLabels)
		{
			Node temp = new Node();
			temp.trainingLabels = CloneLabelList(trainingLabels);
			
			double maxInfoGain = -1000;
			Word maxInfo = null;
			foreach(Word w in TotalWords)
			{
				double tempInfoGain = InformationGain(w, trainingLabels);
				//Multiplying with number of labels
				//tempInfoGain = tempInfoGain * trainingLabels.Count();
				if(tempInfoGain > maxInfoGain)				
				{
					maxInfo = w;
					maxInfoGain = tempInfoGain;
				}
			}
			
			temp.word = maxInfo;
			temp.InformationGain = maxInfoGain;
			temp.WithWord = null;
			temp.WithoutWord = null;
			temp.isLeafNode = true;
			
			int label1 = trainingLabels.FindAll(p=>p.label == 1).Count;
			int label2 = trainingLabels.FindAll(p=>p.label == 2).Count;
			
			temp.classLabel = (label1 > label2) ? 1 : 2;
			
			temp.label1Count = label1;
			temp.label2Count = label2;
			
			return temp;
		}
		
		List<Label> CloneLabelList(List<Label> examples)
		{
			//cloning the list
			List<Label> tempLabels = new List<Label>();
			foreach(Label l in examples)
				tempLabels.Add(new Label(){ label = l.label, documentId = l.documentId});
			
			return tempLabels;
		}
		
		double TestAccuracyOfTree(Node root, List<Label> test, List<Document> docs)
		{
			//Testing the Test Set
			int failedCount = 0;
			int passedCount = 0;
			foreach(Label testLabel in test)
			{
				Node t = root;
				while(!t.isLeafNode)
				{
					if(docs.Exists(p=>p.documentId == testLabel.documentId && p.words.Contains(t.word.wordId)))
						t = t.WithWord;
					else
						t = t.WithoutWord;
				}
				if(t.classLabel != testLabel.label)
					failedCount++;
				else
					passedCount++;
			}
			
			return (double)passedCount/(double)(passedCount+failedCount);
		}
		
		public void PrepareWekaData(string fileName, List<Document> docs, List<Label> labels)
		{
			
//			string filePath = @"C:\test.csv";  
//			string delimiter = ",";  
//3	 
//4	            string[][] output = new string[][]{  
//5	                new string[]{"Col 1 Row 1", "Col 2 Row 1", "Col 3 Row 1"},  
//6	                new string[]{"Col1 Row 2", "Col2 Row 2", "Col3 Row 2"}  
//7	            };  
//8	            int length = output.GetLength(0);  
//9	            StringBuilder sb = new StringBuilder();  
//10	            for (int index = 0; index < length; index++)  
//11	                sb.AppendLine(string.Join(delimiter, output[index]));  
//12	 
//13	            File.WriteAllText(filePath, sb.ToString()); 
			
			
			//StreamWriter sw = new StreamWriter(fileName);	//
			StringBuilder sb = new StringBuilder();
			
			//1st line
			sb.Append("documentId,");
			for(int i=0; i< TotalWords.Count; i++)
				sb.Append(TotalWords[i].word + ",");
			sb.AppendLine("Label");
				
			foreach(Label l in labels)	//Training Label
			{
				Document doc = docs.Find(d => d.documentId == l.documentId);
				if(doc!=null)
				{
					sb.Append(doc.documentId + ",");
					for(int i=0; i<TotalWords.Count; i++)
					{
						if(doc.words.Contains(TotalWords[i].wordId))
							sb.Append("1,");
						else
							sb.Append("0,");
					}
					sb.AppendLine(l.label.ToString());
				}
			}
			//sw.Flush();
			//sw.Close();
			File.WriteAllText(fileName, sb.ToString());
		}
	}
	
	class Document
	{
		public int documentId;
		public List<int> words;
	}
	
	class Word
	{
		public int wordId;
		public string word;
	}
	
	class Label
	{
		public int documentId;
		public int label;	//Label is either 1 or 2 -- 1=alt.atheism or 2=comp.graphics
	}
	
	class Node
	{
		public Word word;
		public double InformationGain;
		public List<Label> trainingLabels;
		
		public Node WithWord;	//Child1
		public Node WithoutWord;	//Child2
		
		public bool isLeafNode;
		public int classLabel;
		
		public int label1Count;
		public int label2Count;
	}
}
