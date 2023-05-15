const express = require("express");
const router = express.Router();
const fs = require("fs");
const {
  getEAs,
  getTopic,
  getArticle,
  getNodes,
  getEdges,
  getChart,
} = require("../helpers/helpers");
const { fileInfoJson } = require("../environments");
const keyword = "by_omission";
const rawdata = fs.readFileSync(fileInfoJson);
const files = JSON.parse(rawdata);

router.get("/", (req, res) => {
  res.render(keyword, { data: files });
});

router.get("/:topic_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let articles = [];
  if (topic) articles = topic["articles"];
  res.render(keyword, {
    topic: topic,
    topic_id: req.params.topic_id,
    articles: articles,
    nodes: getNodes(topic),
    edges: getEdges(topic),
  });
});

router.get("/:topic_id/:article_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let queried_article = getArticle(topic, req.params.article_id);
  let results = null;
  if (queried_article) {
    results = JSON.parse(
      fs.readFileSync("./public/" + queried_article["analyzed"])
    );
  }
  let ea = getEAs(results, "by_omission");
  let barChart = getChart(ea, "non_reused");
  res.render(keyword, {
    topic: topic,
    topic_id: req.params.topic_id,
    article_id: req.params.article_id,
    articles: topic["articles"],
    is_biased: results["is_biased"],
    queried_article: queried_article,
    earlier_articles: ea,
    by_omission: results[keyword],
    stackedBarData: barChart["bar"],
    stackedBarLabels: barChart["labels"],
    stackedBarTickColors: barChart["tickColors"],
    nodes: getNodes(topic),
    edges: getEdges(topic),
    is_earliest: results["is_earliest"],
  });
});

router.get("/:topic_id/:article_id/:earlier_article_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let queried_article = getArticle(topic, req.params.article_id);
  let results = null;
  if (queried_article) {
    results = JSON.parse(
      fs.readFileSync("./public/" + queried_article["analyzed"])
    );
  }
  let EAs = getEAs(results, "by_omission");
  let barChart = getChart(EAs, "non_reused");
  if (results && results["earlier_articles"]) {
    EAs = results["earlier_articles"].filter((f) => {
      return results["by_omission"]["omitted_articles"].includes(
        parseInt(f["ea_id"])
      );
    });
  }
  let earlier_article = EAs.find((f) => {
    return parseInt(f["ea_id"]) === parseInt(req.params.earlier_article_id);
  });
  res.render(keyword, {
    topic: topic,
    topic_id: req.params.topic_id,
    article_id: req.params.article_id,
    articles: topic["articles"],
    is_biased: results["is_biased"],
    queried_article: queried_article,
    earlier_articles: EAs,
    earlier_article_details: earlier_article,
    non_reused_details: earlier_article["ea_non_reused_details"],
    by_omission: results[keyword],
    stackedBarData: barChart["bar"],
    stackedBarLabels: barChart["labels"],
    stackedBarTickColors: barChart["tickColors"],
    nodes: getNodes(topic),
    edges: getEdges(topic),
    is_earliest: results["is_earliest"],
  });
});

module.exports = router;
