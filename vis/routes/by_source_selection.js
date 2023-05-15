const express = require("express");
const fs = require("fs");
const router = express.Router();
const {
  getEAs,
  getTopic,
  getArticle,
  getNodes,
  getEdges,
  getChart,
} = require("../helpers/helpers");
const keyword = "by_source_selection";
const { fileInfoJson } = require("../environments");
const rawdata = fs.readFileSync(fileInfoJson);
const files = JSON.parse(rawdata);

router.get("/", (req, res) => {
  res.render(keyword, { data: files });
});

/////////////////////////////////////////////////////////////////////

router.get("/:topic_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let articles = [];
  if (topic) articles = topic["articles"];
  res.render("by_source_selection", {
    topic: topic,
    topic_id: req.params.topic_id,
    articles: articles,
    nodes: getNodes(topic),
    edges: getEdges(topic),
  });
});

/////////////////////////////////////////////////////////////////////

router.get("/:topic_id/:article_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let queried_article = getArticle(topic, req.params.article_id);
  let results = null;
  if (queried_article) {
    results = JSON.parse(
      fs.readFileSync("./public/" + queried_article["analyzed"])
    );
  }
  let EAs = getEAs(results, "by_source_selection");
  let barChart = getChart(EAs, "reused");
  res.render("by_source_selection", {
    topic: topic,
    topic_id: req.params.topic_id,
    article_id: req.params.article_id,
    articles: topic["articles"],
    is_biased: results["is_biased"],
    queried_article: queried_article,
    earlier_articles: EAs,
    by_source_selection: results[keyword],
    stackedBarData: barChart["bar"],
    stackedBarLabels: barChart["labels"],
    stackedBarTickColors: barChart["tickColors"],
    nodes: getNodes(topic),
    edges: getEdges(topic),
    is_earliest: results["is_earliest"],
  });
});

/////////////////////////////////////////////////////////////////////

router.get("/:topic_id/:article_id/:earlier_article_id", (req, res) => {
  let topic = getTopic(files, req.params.topic_id);
  let queried_article = getArticle(topic, req.params.article_id);
  let results = null;
  if (queried_article) {
    results = JSON.parse(
      fs.readFileSync("./public/" + queried_article["analyzed"])
    );
  }
  let EAs = getEAs(results, "by_source_selection");
  let earlier_article = EAs.find((f) => {
    return parseInt(f["ea_id"]) === parseInt(req.params.earlier_article_id);
  });
  let barChart = getChart(EAs, "reused");
  res.render("by_source_selection", {
    topic: topic,
    topic_id: req.params.topic_id,
    article_id: req.params.article_id,
    articles: topic["articles"],
    queried_article: queried_article,
    earlier_articles: EAs,
    earlier_article_details: earlier_article,
    is_biased: results["is_biased"],
    earlier_article_id: req.params.earlier_article_id,
    aic_label: earlier_article["article_label"],
    reused_ratio: earlier_article["ea_reused_ratio"],
    label_ratio: earlier_article["ea_reused_label_percent"],
    is_biased_to:
      earlier_article["is_biased_to"] !== ""
        ? earlier_article["is_biased_to"]
        : "No",
    ea: earlier_article ? earlier_article["ea_reused_details"] : [],
    by_source_selection: results[keyword],
    stackedBarData: barChart["bar"],
    stackedBarLabels: barChart["labels"],
    stackedBarTickColors: barChart["tickColors"],
    nodes: getNodes(topic),
    edges: getEdges(topic),
    is_earliest: results["is_earliest"],
  });
});

module.exports = router;
