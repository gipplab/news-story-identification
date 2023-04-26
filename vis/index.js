"use strict";
const express = require("express");
const app = express();
const port = 80;
const fs = require("fs");
app.use(express.static("public"));
app.set("view engine", "ejs");
app.set("views", "./templates");

app.get("/", (req, res) => {
  res.render("index");
});

/*
 *  BY SOURCE SELECTION
 */
app.get(
  "/by_source_selection/:topic_id/:article_id/:earlier_article_id",
  (req, res) => {
    let rawdata = fs.readFileSync(
      "../data/ground.news/by_source_selection/files_info.json"
    );
    let files = JSON.parse(rawdata);
    let topic = files.find((f) => {
      return f["topic_id"] === req.params.topic_id;
    });
    let queried_article = topic["articles"].find((f) => {
      return parseInt(f["article_id"]) === parseInt(req.params.article_id);
    });
    let results = null;
    if (queried_article) {
      results = JSON.parse(
        fs.readFileSync(__dirname + "/public/" + queried_article["analyzed"])
      );
    }
    let ea = [];
    if (results && results["earlier_articles"]) {
      ea = results["earlier_articles"].filter((f) => {
        return results["biased_by_these_sources"].includes(
          parseInt(f["article_id"])
        );
      });
    }
    let earlier_article = ea.find((f) => {
      return (
        parseInt(f["article_id"]) === parseInt(req.params.earlier_article_id)
      );
    });
    res.render("by_source_selection", {
      topic: topic,
      topic_id: req.params.topic_id,
      article_id: req.params.article_id,
      articles: topic["articles"],
      queried_article: queried_article,
      earlier_articles: ea,
      is_biased: results["is_biased"],
      earlier_article_id: req.params.earlier_article_id,
      aic_label: earlier_article["article_label"],
      reused_ratio: earlier_article["reused_ratio"],
      label_ratio: earlier_article["reused_percentage"],
      is_biased_to:
        earlier_article["is_biased_to"] !== ""
          ? earlier_article["is_biased_to"]
          : "No",
      ea: earlier_article ? earlier_article["reused_details"] : [],
    });
  }
);

app.get("/by_source_selection/:topic_id/:article_id", (req, res) => {
  let rawdata = fs.readFileSync(
    "../data/ground.news/by_source_selection/files_info.json"
  );
  let files = JSON.parse(rawdata);
  let topic = files.find((f) => {
    return f["topic_id"] === req.params.topic_id;
  });
  let queried_article = topic["articles"].find((f) => {
    return parseInt(f["article_id"]) === parseInt(req.params.article_id);
  });
  let results = null;
  if (queried_article) {
    results = JSON.parse(
      fs.readFileSync(__dirname + "/public/" + queried_article["analyzed"])
    );
  }
  let ea = [];
  if (results && results["earlier_articles"]) {
    ea = results["earlier_articles"].filter((f) => {
      return results["biased_by_these_sources"].includes(
        parseInt(f["article_id"])
      );
    });
  }
  console.log(results);
  res.render("by_source_selection", {
    topic: topic,
    topic_id: req.params.topic_id,
    article_id: req.params.article_id,
    articles: topic["articles"],
    is_biased: results["is_biased"],
    queried_article: queried_article,
    earlier_articles: ea,
  });
});

app.get("/by_source_selection/:topic_id", (req, res) => {
  let rawdata = fs.readFileSync(
    "../data/ground.news/by_source_selection/files_info.json"
  );
  let files = JSON.parse(rawdata);
  let topic = files.find((f) => {
    return f["topic_id"] === req.params.topic_id;
  });
  let articles = [];
  if (topic) articles = topic["articles"];
  res.render("by_source_selection", {
    topic: topic,
    topic_id: req.params.topic_id,
    articles: articles,
  });
});

app.get("/by_source_selection/", (req, res) => {
  let rawdata = fs.readFileSync(
    "../data/ground.news/by_source_selection/files_info.json"
  );
  let files = JSON.parse(rawdata);
  res.render("by_source_selection", { data: files });
});

/*
 *  BY COMMISSION
 */

app.get("/by_commission/", (req, res) => {
  // let rawdata = fs.readFileSync(
  //   "../data/ground.news/by_source_selection/files_info.json"
  // );
  // let files = JSON.parse(rawdata);
  let files = [];
  res.render("by_commission", { data: files });
});

/*
 *  BY OMISSION
 */

app.listen(port, () => {
  console.log(`Media bias analysis - Visualization`);
});
