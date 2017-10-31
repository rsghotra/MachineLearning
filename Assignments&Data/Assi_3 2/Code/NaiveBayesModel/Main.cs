using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NaiveBayesModel
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			BayesModel bm = new BayesModel();
			bm.ComputeParameters();
			
			Console.WriteLine("Training Accuracy: " + bm.ComputeAccuracy(bm.trainingSet, bm.trainingDocs).ToString());
			Console.WriteLine("Test Accuracy: " + bm.ComputeAccuracy(bm.testSet, bm.testDocs).ToString());

            Console.ReadLine();
		}
	}
	
	class BayesModel
	{
		//Lists used to store the data from files
		public List<Document> trainingDocs = new List<Document>();
		public List<Document> testDocs = new List<Document>();
		public List<Label> trainingSet = new List<Label>();
		public List<Label> testSet = new List<Label>();
		List<Word> TotalWords = new List<Word>();
		
		List<Parameter> paramsLabel1 = new List<Parameter>();
		List<Parameter> paramsLabel2 = new List<Parameter>();
		
		public BayesModel()
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
		
		public void ComputeParameters()	//Using Maximum likelihood and Laplacian Smoothing
		{
			paramsLabel1 = new List<Parameter>();
			paramsLabel2 = new List<Parameter>();
			
			List<WordFeature> wordFeatures = new List<WordFeature>();
			
			foreach(Word w in TotalWords)
			{
				//Class Label = 1, each word 'Wi' is true
				//Pr(Word = true| Label=1) from training set
				Parameter temp1 = new Parameter();
				temp1.word = new Word(){ wordId = w.wordId, word = w.word};			
				int wordPresentAndLabel1 = trainingSet.FindAll(l => l.label == 1 && trainingDocs.Exists(d => d.documentId == l.documentId && d.words.Contains(w.wordId))).Count;
				int wordNotPresentAndLabel1 = trainingSet.FindAll(l => l.label == 1 && trainingDocs.Exists(d => d.documentId == l.documentId && !d.words.Contains(w.wordId))).Count;
				
				temp1.probablility = (double)(wordPresentAndLabel1 + 1.0)/(double)(wordPresentAndLabel1 + wordNotPresentAndLabel1 + 2.0);
				paramsLabel1.Add(temp1);
						
				//Class Label = 2, each word 'Wi' is true
				//Pr(Word = true| Label=2) from training set
				Parameter temp2 = new Parameter();
				temp2.word = new Word(){ wordId = w.wordId, word = w.word};			
				int wordPresentAndLabel2 = trainingSet.FindAll(l => l.label == 2 && trainingDocs.Exists(d => d.documentId == l.documentId && d.words.Contains(w.wordId))).Count;
				int wordNotPresentAndLabel2 = trainingSet.FindAll(l => l.label == 2 && trainingDocs.Exists(d => d.documentId == l.documentId && !d.words.Contains(w.wordId))).Count;
				
				temp2.probablility = (double)(wordPresentAndLabel2 + 1.0)/(double)(wordPresentAndLabel2 + wordNotPresentAndLabel2 + 2.0);
				paramsLabel2.Add(temp2);
				
				WordFeature wf = new WordFeature();
				wf.word = w;
				wf.val = Math.Abs( Math.Log(temp1.probablility) - Math.Log(temp2.probablility) );
				wordFeatures.Add(wf);
			}
			
			wordFeatures.Sort((x, y) => x.val.CompareTo(y.val));
			wordFeatures.Reverse();
			
			for(int i=0;i<10;i++)
				Console.WriteLine(wordFeatures[i].word.word);
		}
		
		public double ComputeAccuracy(List<Label> labels, List<Document> docs)
		{
			int success = 0;
			int failure = 0;
			
			int label1 = labels.FindAll(l => l.label == 1).Count;
			int label2 = labels.FindAll(l => l.label == 2).Count;
			
			foreach(Label lab in labels)
			{
				Document d = docs.Find(p=>p.documentId == lab.documentId);
				
				//Pr(Label = 1|W1,W2,W3,.....,Wn) Wi's in d
				double prob1 = (double)(label1)/(label1+label2);
				double prob2 = (double)(label2)/(label1+label2);
				
				double log1 = Math.Log(prob1);
				double log2 = Math.Log(prob2);	//Taking logarithms to prevent underflow
				
				foreach(Word wd in TotalWords)    //int wordId in d.words)
				{
					Parameter p1 = paramsLabel1.Find(w => w.word.wordId == wd.wordId);
                    if(d.words.Contains(wd.wordId))
					    log1 = log1 + Math.Log(p1.probablility);
                    else
                        log1 = log1 + Math.Log(1-p1.probablility);
				
					//Pr(Label = 2|W1,W2,W3,.....,Wn) Wi's in d
					Parameter p2 = paramsLabel2.Find(w => w.word.wordId == wd.wordId);
                    if (d.words.Contains(wd.wordId))
					    log2 = log2 + Math.Log(p2.probablility);
                    else
                        log2 = log2 + Math.Log(1 - p2.probablility);
				}
				
				int label = -1;
				if(log1 > log2)
					label = 1;
				else
					label = 2;
				
				if(label == lab.label)
					success++;
				else
					failure++;
			}
			
			return (double)success/(double)(success+failure);
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
	
	class Parameter
	{
		public Word word;
		public double probablility;
	}
	
	class WordFeature
	{
		public Word word;
		public double val;
	}
}
