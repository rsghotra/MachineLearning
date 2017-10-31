using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace ReadData
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			StreamReader sr = new StreamReader(@"/u3/rguttikonda/Desktop/Latest/pts2");
			string temp1 = sr.ReadLine();
			string temp2 = sr.ReadLine();
			
			Dictionary<int,double> dict1 = new Dictionary<int, double>();
			Dictionary<int,double> dict2 = new Dictionary<int, double>();
			while(temp1 != null && temp2!=null)
			{
				List<string> tempVals = temp1.Split(' ').ToList();
				tempVals.RemoveAll(s => s==null || s=="");
				int x1 = int.Parse(tempVals[0]);
				double y1 = double.Parse(tempVals[1]);
				
				tempVals = temp2.Split(' ').ToList();
				tempVals.RemoveAll(s => s==null || s=="");
				int x2 = int.Parse(tempVals[0]);
				double y2 = double.Parse(tempVals[1]);
				
				
				dict1.Add(x1,y1);
				dict2.Add(x2,y2);
				
				temp1 = sr.ReadLine();
				if(temp1!=null)
					temp2 = sr.ReadLine();
				else
					break;
			}
			sr.Close();
			
			for(int i=1;i<=dict1.Count;i++)
				Console.WriteLine(i.ToString() + "  " + dict1[i].ToString());
			for(int i=1;i<=dict2.Count;i++)
				Console.WriteLine(i.ToString() + "  " + dict2[i].ToString());
		}
	}
}
