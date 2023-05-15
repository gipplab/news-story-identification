exports.getEAs = (results, type) => {
  // type = "by_omission" | "by_commission" | "by_source_selection"
  let ea = [];
  if (type && results) {
    if (results && results["earlier_articles"]) {
      ea = results["earlier_articles"].filter((f) => {
        if (type === "by_omission")
          return results["by_omission"]["omitted_articles"].includes(
            parseInt(f["ea_id"])
          );
        else
          return results["by_source_selection"]["committed_articles"].includes(
            parseInt(f["ea_id"])
          );
      });
    }
  }

  return ea;
};

exports.getTopic = (files, topic_id) => {
  return files.find((f) => {
    return f["topic_id"] === topic_id;
  });
};

exports.getArticle = (topic, article_id) => {
  return topic["articles"].find((f) => {
    return parseInt(f["article_id"]) === parseInt(article_id);
  });
};

exports.getNodes = (topic) => {
  let graph = topic.graph;
  let nodes = graph.nodes.map((i) => {
    return {
      // ...i,
      id: i.id,
      label: String(i.label),
      color: i.color,
      font: { color: "white", face: "arial" },
    };
  });
  return nodes;
};

exports.getEdges = (topic) => {
  let graph = topic.graph;
  let edges = graph.edges.map((i) => {
    return {
      // ...i,
      from: i.from,
      to: i.to,
      label: i.label,
      font: { align: "bottom" },
      arrows: "to",
    };
  });
  return edges;
};

exports.getChart = (ea, type = "reused") => {
  let plagType = type === "reused" ? "ea_reused_label" : "ea_non_reused_label";

  let left = [];
  let center = [];
  let right = [];
  let labels = [];
  let tickColors = [];
  ea.forEach((a) => {
    left.push(String(a[plagType]["LEFT"]));
    center.push(String(a[plagType]["CENTER"]));
    right.push(String(a[plagType]["RIGHT"]));
    labels.push(String(a["ea_id"]));
    tickColors.push(
      a["ea_label"] === "LEFT"
        ? "red"
        : a["ea_label"] === "CENTER"
        ? "gray"
        : "blue"
    );
  });

  let stackedBar = [
    { label: "LEFT", data: left, backgroundColor: "red" },
    { label: "CENTER", data: center, backgroundColor: "gray" },
    { label: "RIGHT", data: right, backgroundColor: "blue" },
  ];

  return {
    bar: stackedBar,
    labels: labels,
    tickColors: tickColors,
  };
};
