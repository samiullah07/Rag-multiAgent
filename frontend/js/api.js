const API_ROOT = "/api/v1";

// Exported functions
const queryMultiAgent = async (options) => {
  return await fetch(`${API_ROOT}/query/multi-agent`, {
    method: "POST",
    body: JSON.stringify(options),
    headers: { "Content-Type": "application/json" }
  });
};

const queryBaseline = async (options) => {
  return await fetch(`${API_ROOT}/query/baseline`, {
    method: "POST",
    body: JSON.stringify(options),
    headers: { "Content-Type": "application/json" }
  });
};

const queryStream = async (options) => {
  const reader = new TextDecoderStream().reader;
  const response = await fetch(`${API_ROOT}/query/stream`, {
    method: "POST",
    body: JSON.stringify(options),
    headers: { "Content-Type": "application/json" }
  });

  const stream = await response.json();
  const eventSource = await new EventSource(`${API_ROOT}/query/stream`);
  // Simplified for example - actual implementation needs event handling
  return new Promise((resolve, reject) => {
    let result = {
      type: " initializing",
      answer: "",
      sources: [],
      conflict_info: null
    };
    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "token") {
          result.answer += data.value;
          result.type = "token";
        } else if (data.type === "done") {
          result.answer += data.value;
          result.type = "done";
          resolve(result);
        }
      } catch (err) {
        reject(err);
      }
    };
  });
};

// Add other endpoints (eval/results, experiments) as needed
