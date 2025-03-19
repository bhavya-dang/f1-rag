import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import "dotenv/config";

type SimilarityMetric = "cosine" | "dot_product" | "euclidean";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_ENDPOINT_URL,
  ASTRA_DB_TOKEN,
  OPEN_AI_KEY,
} = process.env;

const openai = new OpenAI({
  apiKey: OPEN_AI_KEY,
});

const f1Data = [
  "https://www.formula1.com/",
  "https://www.formula1.com/en/results.html/2024/drivers.html",
  "https://www.formula1.com/en/results.html/2024/constructors.html",
  "https://www.formula1.com/en/results.html/2024/races.html",
  "https://www.formula1.com/en/results.html/2024/fastest-laps.html",
  "https://www.formula1.com/en/results.html/2024/qualifying.html",
  "https://www.formula1.com/en/results.html/2024/sprint.html",
  "https://en.wikipedia.org/wiki/Formula_One",
  "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship",
  "https://evrimagaci.org/tpg/2025-formula-1-season-kicks-off-with-thrilling-chinese-grand-prix-269074",
  "https://www.formula1.com/en/latest",
];

const client = new DataAPIClient(ASTRA_DB_TOKEN);

const db = client.db(ASTRA_DB_ENDPOINT_URL, {
  namespace: ASTRA_DB_NAMESPACE,
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const createCollection = async (
  similarityMetric: SimilarityMetric = "dot_product"
) => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 1536,
      metric: similarityMetric,
    },
  });
  console.log(res);
};

const loadData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);

  f1Data.forEach(async (url: string) => {
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);
    chunks.forEach(async (chunk) => {
      const embedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: chunk,
        encoding_format: "float",
      });
      const vector = embedding.data[0].embedding;
      const res = await collection.insertOne({
        $vector: vector,
        text: chunk,
      });
      console.log(res);
    });
  });
};

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });

  return (await loader.scrape())?.replace(/<[^>]*>?/gm, "");
};

console.log("Seeding database...");

createCollection().then(() => loadData());
