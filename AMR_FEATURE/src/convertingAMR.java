
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONObject;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

public class convertingAMR {
	private Map<String, Set<String>> map;

	/*initialize from joints.txt, which is used for greedily connecting phrases
	 * e.g. make-up
	 * extracted from AMRPropBank and training set
	 * */
	public convertingAMR(String file) {
		map = new HashMap<String, Set<String>>();
		Set<String> tmp;

		try (FileInputStream fis = new FileInputStream(file);
				BufferedReader br = new BufferedReader(new InputStreamReader(fis, "UTF-8"));) {

			String line;
			String[] pair;
			while ((line = br.readLine()) != null) {
				pair = line.split(" ");
				String past = "";
				for (int i = 0; i < pair.length - 1; i++) {
					past += pair[i] + " ";
					tmp = map.getOrDefault(past.trim(), new HashSet<String>());
					tmp.add(pair[i + 1]);
					map.put(past.trim().replace(" ", "-"), tmp);

				}

			}
		//	System.out.println(map.toString());

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static List<String> folderToFilesPath(String folder, String suffix) {
		List<String> results = new ArrayList<String>();

		File[] files = new File(folder).listFiles();
		// If this pathname does not denote a directory, then listFiles() returns null.

		for (File file : files) {
			if (file.isFile() && file.getName().endsWith(suffix)) {
				results.add(folder + file.getName());
			}
		}
		return results;
	}

	public void featureExtract(String file) {

		// build pipeline
		StanfordCoreNLP pipeline = new StanfordCoreNLP(
				PropertiesUtils.asProperties("annotators", "tokenize,ssplit,pos,lemma,ner", "tokenize.options",
						"splitHyphenated=true", 
						"tokenize.whitespace", "true",//start with tokenized file 
						"ssplit.isOneSentence",  //ignore multi-sentence construction
						"true", "tokenize.language", "en"));
		
		String[] name = file.split("/");

		String line = "";
		try (FileInputStream fis = new FileInputStream(file);
				BufferedReader br = new BufferedReader(new InputStreamReader(fis, "UTF-8"));) {

			System.out.println(name[name.length - 1]);
			int n = 0;
			int changed = 0;
			List<JSONObject> obs = new ArrayList<JSONObject>();
			line = br.readLine();
			while (line != null && !line.trim().isEmpty()) {
				// if (n % 2 == 0)
				// System.out.println(n+"\n"+line);
				n++;
				JSONObject obj = new JSONObject();
				StringBuilder pre = new StringBuilder();
				while (!line.startsWith("# ::tok ") && !line.startsWith("# ::snt ")) {
					pre.append(line + "\n");
					line = br.readLine();
				}
				obj.put("pre", pre.toString());
				
				//build a sentence without buggy texts....
				String snt = line.replace("# ::tok ", "").replace("# ::snt ", "");
				snt = snt.replaceAll("\\.{2,}", "").replaceAll("  ", " ");
				snt = snt.replace("  ", " ").replace("  ", " ").replace("\n", "");
				snt = snt.replaceAll("\"", " \" ");
				snt = snt.replaceAll("\\(", " \\( ");
				snt = snt.replaceAll("\\)", " \\) ");
				snt = snt.replaceAll("@-@", "-").replaceAll(" @:@ ", ":");
				obj.put("snt", snt);

				//initial feature extraction and connecting of phrase
				HashMap<String, LinkedList<String>> data = extractSentence(obj, pipeline,true);
				//connects number
				changed += post_procee_number(data);
				//connects ner, mainly due to "-" and "'s" construction in AMR NER
				changed += post_procee_ner(data);
				obj.put("ner", data.get("ner"));
				obj.put("lem", data.get("lem"));
				obj.put("tok", data.get("tok"));
				obj.put("pos", data.get("pos"));
				obs.add(obj);

				if (obs.size() % 500 == 0) {
					System.out.println(obs.size() + " " + name[name.length - 1]);
					obj.keys().forEachRemaining(k -> {
						System.out.println(k + ": " + obj.get(k));
					});
				}

				//read remaining e.g. AMR graph
				StringBuilder post = new StringBuilder();
				line = br.readLine();
				while (line != null && !line.trim().isEmpty()) {
					post.append(line + "\n");
					line = br.readLine();
				}

				obj.put("post", post.toString());

				while (line != null && line.trim().isEmpty()) {
					line = br.readLine();
				}

			}
			System.out.println("\n" + name[name.length - 1] + " done. Total sentences: " + obs.size() + "\n");
			System.out.println("\n" + changed + " changed." + "\n");
			String out = obs.stream().map(obj -> writeObject(obj)).collect(Collectors.joining("\n"));
			Files.write(Paths.get(file.replaceAll(".txt(_[a-z]*)*", ".txt_pre_processed")), out.getBytes());

		} catch (IOException e) {
			e.printStackTrace();
		} catch (NullPointerException e) {
			System.out.println(file + "  null pointer??");
			System.out.println(line + "  null pointer??");
			e.printStackTrace();
		}
	}
	
	//same as featureExtract, but have sentence only
	public void featureExtractSentenceOnly(String file) {

		StanfordCoreNLP pipeline = new StanfordCoreNLP(
				PropertiesUtils.asProperties("annotators", "tokenize,ssplit,pos,lemma,ner", "tokenize.options",
						"splitHyphenated=true", "tokenize.whitespace", "true",
						"ssplit.isOneSentence", "true", "tokenize.language", "en"));
		String[] name = file.split("/");

		String line = "";
		try (FileInputStream fis = new FileInputStream(file);
				BufferedReader br = new BufferedReader(new InputStreamReader(fis, "UTF-8"));) {

			System.out.println(name[name.length - 1]);
			int n = 0;
			int changed = 0;
			List<JSONObject> obs = new ArrayList<JSONObject>();
			line = br.readLine();
			while (line != null && !line.trim().isEmpty()) {
				// if (n % 2 == 0)
				// System.out.println(n+"\n"+line);
				n++;
				JSONObject obj = new JSONObject();
				StringBuilder pre = new StringBuilder();
				obj.put("pre", pre.toString());
				
				//build a sentence without buggy texts....
				String snt = line.replace("# ::tok ", "").replace("# ::snt ", "");
		/*		if (snt.startsWith("the ones who are suffering are the ordinary people :")) {
					System.out.println("!!!!\n"+snt+"\n!!!!");
				}
				snt = snt.replaceAll("\\.{2,}", "").replaceAll("  ", " ");
				if (snt.startsWith("the ones who are suffering are the ordinary people :")) {
					System.out.println("!!!!\n"+snt+"\n!!!!");
				}
				snt = snt.replace("  ", " ").replace("  ", " ").replace("\n", "");
				snt = snt.replaceAll("\"", " \" ");
				snt = snt.replaceAll("\\(", " \\( ");
				snt = snt.replaceAll("\\)", " \\) ");
				snt = snt.replaceAll("@-@", "-").replaceAll(" @:@ ", ":");*/
				obj.put("snt", snt);
					
				//feature extraction and connecting of phrase, no change of tokenization
				HashMap<String, LinkedList<String>> data = extractSentence(obj, pipeline,false);
				if (snt.startsWith("the ones who are suffering are the ordinary people :")) {
					System.out.println("!!!!\n"+snt+"\n!!!!");
					System.out.println( data.get("tok"));
				}
				obj.put("ner", data.get("ner"));
				obj.put("lem", data.get("lem"));
				obj.put("tok", data.get("tok"));
				obj.put("pos", data.get("pos"));
				obs.add(obj);

				if (obs.size() % 500 == 0) {
					System.out.println(obs.size() + " " + name[name.length - 1]);
					obj.keys().forEachRemaining(k -> {
						System.out.println(k + ": " + obj.get(k));
					});
				}

				StringBuilder post = new StringBuilder();
				obj.put("post", post.toString());

				line = br.readLine();
				while (line != null && line.trim().isEmpty()) {
					line = br.readLine();
				}

			}
			System.out.println("\n" + name[name.length - 1] + " done. Total sentences: " + obs.size() + "\n");
			System.out.println("\n" + changed + " changed." + "\n");
			String out = obs.stream().map(obj -> writeObject(obj)).collect(Collectors.joining("\n"));
			Files.write(Paths.get(file.replaceAll(".txt(_[a-z]*)*", ".txt_processed")), out.getBytes());

		} catch (IOException e) {
			e.printStackTrace();
		} catch (NullPointerException e) {
			System.out.println(file + "  null pointer??");
			System.out.println(line + "  null pointer??");
			e.printStackTrace();
		}
	}

	public String jsonArrayToWhiteSpaceString(JSONArray arr) {
		String out = "";
		for (Object s : arr) {
			out += "\t" + s.toString();
		}
		return out.substring(1);
	}

	// output data to txt
	public String writeObject(JSONObject obj) {
		StringBuilder out = new StringBuilder();
		if (!obj.getString("pre").trim().isEmpty())
			out.append(obj.getString("pre"));
		String tok = jsonArrayToWhiteSpaceString(obj.getJSONArray("tok"));
		out.append("# ::tok\t" + tok + "\n");
		String lemma = jsonArrayToWhiteSpaceString(obj.getJSONArray("lem"));
		out.append("# ::lem\t" + lemma + "\n");
		String pos = jsonArrayToWhiteSpaceString(obj.getJSONArray("pos"));
		out.append("# ::pos\t" + pos + "\n");
		String ner = jsonArrayToWhiteSpaceString(obj.getJSONArray("ner"));
		out.append("# ::ner\t" + ner + "\n");

		assert tok.split(" ").length == lemma.split(" ").length;
		assert tok.split(" ").length == pos.split(" ").length;
		assert tok.split(" ").length == ner.split(" ").length;

		if (!obj.getString("post").trim().isEmpty())
			out.append(obj.getString("post") + "\n");

		return out.toString();
	}

	public volatile int positive = 0;
	public volatile int truth = 0;
	public volatile int truth_positive = 0;

	public HashMap<String, LinkedList<String>> extractSentence(JSONObject obj, StanfordCoreNLP pipeline,boolean retoken) {
		String text = obj.getString("snt");
		obj.put("snt", text);
		// create an empty Annotation just with the given text

		Annotation sent = new Annotation(text);
		HashMap<String, LinkedList<String>> data = new HashMap<String, LinkedList<String>>();
		// run all Annotators on this text
		pipeline.annotate(sent);
		LinkedList<String> lemma = new LinkedList<String>();
		LinkedList<String> tok = new LinkedList<String>();
		LinkedList<String> ner = new LinkedList<String>();
		LinkedList<String> pos = new LinkedList<String>();
		String p_l = "";
		String p_s = "";
		String p_n = "";
		String p_p = "";
		int changed = 0;

		List<CoreMap> sentences = sent.get(SentencesAnnotation.class);
		Set<String> tmp = new HashSet<String>();
		for (CoreMap sentence : sentences) {

			for (int i = 0; i < sentence.get(TokensAnnotation.class).size(); i++) {
				CoreLabel token = sentence.get(TokensAnnotation.class).get(i);
				if (retoken  &&tmp.contains(token.get(LemmaAnnotation.class))
						&& (!map.containsKey(lemma.getLast() + "-" + token.get(LemmaAnnotation.class)) //not x-y-z
								|| (i + 1 < sentence.get(TokensAnnotation.class).size() - 1 && 
										map.get(lemma.getLast() + "-" + token.get(LemmaAnnotation.class))
										.contains(sentence.get(TokensAnnotation.class).get(i + 1)
												.get(LemmaAnnotation.class))

								))) {
					p_s = tok.removeLast();
					p_l = lemma.removeLast();
					p_p = pos.removeLast();
					p_n = ner.removeLast();
					changed = 1;
					tok.add(p_s + "-" + token.get(TextAnnotation.class));
					lemma.add(p_l + "-" + token.get(LemmaAnnotation.class).toLowerCase());
					pos.add("COMP");
					ner.add("O");
				} else {

					tok.add(token.get(TextAnnotation.class));
					lemma.add(token.get(LemmaAnnotation.class).toLowerCase());
					pos.add(token.get(PartOfSpeechAnnotation.class));
					if (lemma.get(lemma.size() - 1).contains("www.") || lemma.get(lemma.size() - 1).contains("http"))
						ner.add("URL");
					else
						ner.add(token.get(NamedEntityTagAnnotation.class));

				}
				tmp = map.getOrDefault(lemma.getLast(), new HashSet<String>());
			}

		}
		assert ner.size() == lemma.size() && lemma.size() == tok.size() && tok.size() == pos.size();
		data.put("lem", lemma);
		data.put("tok", tok);
		data.put("pos", pos);
		data.put("ner", ner);
		return data;

	}

	private String[] tobehashed = { "hundred", "thousand", "million", "billion", "trillion", "hundreds", "thousands",
			"millions", "billions", "trillions", "-" };
	private HashSet<String> num_txts = new HashSet<>(Arrays.asList(tobehashed));

	public boolean number_read(String old, String t) {
		return num_txts.contains(t) && !old.equals("-") && !t.equals("-");
	}

	public int post_procee_number(HashMap<String, LinkedList<String>> obj) {
		LinkedList<String> ner_ = (LinkedList<String>) obj.get("ner");
		LinkedList<String> lemma_ = (LinkedList<String>) obj.get("lem");
		LinkedList<String> tok_ = (LinkedList<String>) obj.get("tok");
		LinkedList<String> pos_ = (LinkedList<String>) obj.get("pos");
		String p_l = "";
		String p_t = "";
		String p_n = "";
		String p_p = "";
		LinkedList<String> lemma = new LinkedList<String>();
		LinkedList<String> tok = new LinkedList<String>();
		LinkedList<String> ner = new LinkedList<String>();
		LinkedList<String> pos = new LinkedList<String>();
		int out = 0;
		for (int i = 0; i < lemma_.size(); i++) {
			if (pos.isEmpty() || !pos_.get(i).equals("CD") || (!pos.isEmpty() && !pos.getLast().equals("CD"))
					|| (!number_read(lemma.getLast(), lemma_.get(i)))) {

				lemma.add(lemma_.get(i));
				tok.add(tok_.get(i));
				ner.add(ner_.get(i));
				pos.add(pos_.get(i));
			} else {
				if (lemma.getLast().equals("-")) {
					System.out.println("!!!" + lemma.getLast() + " " + lemma_.get(i));
					System.out.println("!!!" + tok_);
					System.out.println("!!!" + pos_);
				}
				out += 1;
				p_t = tok.removeLast();
				p_l = lemma.removeLast();
				p_p = pos.removeLast();
				p_n = ner.removeLast();

				tok.add(p_t + "," + tok_.get(i));
				lemma.add(p_l + "," + lemma_.get(i));
				pos.add("CD");
				ner.add(p_n);
			}
		}
		obj.put("lem", lemma);
		obj.put("tok", tok);
		obj.put("pos", pos);
		obj.put("ner", ner);
		return out;
	}

	public int post_procee_ner(HashMap<String, LinkedList<String>> obj) {
		LinkedList<String> ner_ = obj.get("ner");
		LinkedList<String> lemma_ = obj.get("lem");
		LinkedList<String> tok_ = obj.get("tok");
		LinkedList<String> pos_ = obj.get("pos");
		String p_l = "";
		String p_t = "";
		String p_n = "";
		String p_p = "";
		LinkedList<String> lemma = new LinkedList<String>();
		LinkedList<String> tok = new LinkedList<String>();
		LinkedList<String> ner = new LinkedList<String>();
		LinkedList<String> pos = new LinkedList<String>();

		Set<String> tmp = new HashSet<String>();
		int out = 0;
		boolean last = false;
		for (int i = 0; i < lemma_.size(); i++) {
			if (( !ner_.get(i).equals("O")) && ( lemma_.get(i).equals("'s") ||lemma_.get(i).equals("-")|| last)
					&& !ner.isEmpty() && ner.getLast().equals(ner_.get(i))) {

				p_t = tok.removeLast();
				p_l = lemma.removeLast();
				p_p = pos.removeLast();
				p_n = ner.removeLast();
				last = lemma_.get(i).equals("-");
				out += 1;
				tok.add(p_t + tok_.get(i));
				lemma.add(p_l + lemma_.get(i));
				pos.add(p_p);
				ner.add(p_n);
			} else {
				last = false;
				lemma.add(lemma_.get(i));
				tok.add(tok_.get(i));
				ner.add(ner_.get(i));
				pos.add(pos_.get(i));
			}
		}
		obj.put("lem", lemma);
		obj.put("tok", tok);
		obj.put("pos", pos);
		obj.put("ner", ner);
		return out;
	}

	public void featureExtractFolder(String folder, String suffix) {
		List<String> files = convertingAMR.folderToFilesPath(folder, suffix);
		files.parallelStream().forEach(file -> featureExtract(file));
	}

	public void featureExtractFolderSentenceOnly(String folder, String suffix) {
		List<String> files = convertingAMR.folderToFilesPath(folder, suffix);
		files.parallelStream().forEach(file -> featureExtractSentenceOnly(file));

	}

	public static void main(String[] args) {
		String home = System.getProperty("user.home");  //change this accordingly

		convertingAMR convetor = new convertingAMR("joints.txt");

		System.out.println("Processing r2");
		System.out.println("Processing Dev");
		convetor.featureExtractFolder(home + "/Data/amr_annotation_r2/data/alignments/split/dev/", "combined.txt_");
		System.out.println("Processing Training");
		convetor.featureExtractFolder(home + "/Data/amr_annotation_r2/data/alignments/split/training/", "combined.txt_");
		System.out.println("Processing Test");
		convetor.featureExtractFolder(home + "/Data/amr_annotation_r2/data/alignments/split/test/", "combined.txt_");



	}

}
