package me.yrutis.javaparsing;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.*;
import java.nio.file.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static java.lang.Thread.sleep;

public class FileListing {
    private final static int numThreads = Runtime.getRuntime().availableProcessors(); //number of threads depends on available processors for optimal performance
    private final static ExecutorService executorService  = Executors.newFixedThreadPool(numThreads);

    private final static JSONArray[] jsonArrays = new JSONArray[numThreads];
    private final static LinkedBlockingQueue<String> linkedBlockingQueue = new LinkedBlockingQueue<String>();

    private final static int LIMIT = 25000;
    private final static AtomicInteger atomicInteger = new AtomicInteger(0);
    private final static String projectFile = "C:\\Users\\yvesr\\Downloads\\evaldata\\test";
    private final static String path = "C:\\Users\\yvesr\\Downloads\\evaldata\\test_parsed\\";


    public static void main(String[] args) throws IOException {
        PathMatcher javaMatcher = FileSystems.getDefault().getPathMatcher("glob:**.java");
        Path start = Paths.get(projectFile);

        // Add all java filesNames to blockingQueue
        try( Stream<Path> pathStream = Files.walk(start) ) {
            pathStream.filter(Files::isRegularFile).filter(javaMatcher::matches).forEach(file ->  putIntoBlockingQueue(file.toString()));
        }

        for (int i = 0; i < numThreads; i++) {
            jsonArrays[i] = new JSONArray();
        }

        class MethodVisitor extends VoidVisitorAdapter {
            private final int id;

            public MethodVisitor(int id) {
                this.id = id;
            }

            public void visit(MethodDeclaration methodDeclaration, Object arg) {
                for (Comment child : methodDeclaration.getAllContainedComments()) {
                    child.remove();
                }

                JSONObject jsonObject = new JSONObject();

                JSONArray params = new JSONArray();
                for(Parameter child : methodDeclaration.getParameters()) {
                    params.add(child.toString());
                }

                jsonObject.put("parameters", params);

                jsonObject.put("Type", methodDeclaration.getType().asString());
                jsonObject.put("methodBody", methodDeclaration.getBody().toString());
                jsonObject.put("methodName", methodDeclaration.getNameAsString());

                FileListing.jsonArrays[id].add(jsonObject);
            }
        }

        for (int i = 0; i < numThreads; i++) {
            final int id = i;
            executorService.submit(new Runnable() {
                @Override
                public void run() {
                    JavaParser javaParser = new JavaParser();
                    FileInputStream fileInputStream;
                    MethodVisitor methodVisitor = new MethodVisitor(id);

                    while(!Thread.currentThread().isInterrupted()) {
                        try {
                            fileInputStream = new FileInputStream(linkedBlockingQueue.take());
                        } catch (IOException | InterruptedException e) {
                            continue;
                        }

                        ParseResult<CompilationUnit> parseResult = javaParser.parse(fileInputStream);

                        if (parseResult.getResult().isPresent()) {
                            methodVisitor.visit(parseResult.getResult().get(), null);
                        }

                        if (LIMIT <= jsonArrays[id].size()) {
                            safe(id);
                        }
                    }
                }
            });
        }

        while(!linkedBlockingQueue.isEmpty()) {
            try {
                System.out.println("remaining java files: " + linkedBlockingQueue.size());
                sleep(5000);
            } catch (InterruptedException ie) {
                System.out.println("main thread go interrupted");
            }
        }
        System.out.println("linkedBlockingQueue is empty");

        for (int i = 0; i < numThreads; i++) {
            if (jsonArrays[i].size() > 0) {
                safe(i);
            }
        }

        executorService.shutdownNow();
        System.exit(0);
    }

    private static void safe(int id) {
        System.out.println("Now saving jsonarray with id " + id + " " + jsonArrays[id].size());
        File file = new File(path + atomicInteger.getAndIncrement() + ".json");
        FileWriter fr = null;
        BufferedWriter br = null;
        PrintWriter pr = null;
        try {
            fr = new FileWriter(file, true);
            br = new BufferedWriter(fr);
            pr = new PrintWriter(br);
            pr.println(jsonArrays[id].toJSONString());
        } catch (IOException e) {
            // ignore
        }

        try {
            pr.close();
            br.close();
            fr.close();
        } catch (NullPointerException | IOException ie) {
            //
        }

        jsonArrays[id] = new JSONArray();
    }

    private static void putIntoBlockingQueue(String filename) {
        boolean put = false;
        while (!put) {
            try {
                linkedBlockingQueue.put(filename);
                put = true;
            } catch (InterruptedException ie) {
                // try again
            }
        }
    }
}
