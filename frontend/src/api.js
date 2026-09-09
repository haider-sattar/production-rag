const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "/api"
).replace(/\/$/, "");
async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  let data = null;

  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    const text = await response.text();
    data = text ? { detail: text } : null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      `Request failed with status ${response.status}`;

    throw new Error(message);
  }

  return data;
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/documents`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

export async function queryDocument(documentId, question) {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document_id: documentId,
      question,
    }),
  });

  return parseResponse(response);
}

export async function deleteDocument(documentId) {
  const response = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
    },
  );

  return parseResponse(response);
}
