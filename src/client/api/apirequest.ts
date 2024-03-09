const API_URL = "http://127.0.0.1:8000/";

class HTTPError extends Error {
  status: number;
  error: string;

  constructor(status: number, error: string) {
    super(error);
    this.status = status;
    this.error = error;
  }
}

const apiRequest = async (method: string, path: string, body: any = null): Promise<any> => {
  let options: {
    method: string;
    headers: { [key: string]: string };
    body: string | FormData | null;
  } = {
    method: method,
    headers: {},
    body: null,
  };

  if (body instanceof FormData) {
    options.body = body;
  } else if (body) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  let uri = API_URL + path;
  let response = await fetch(uri, options);

  if (response.status === 200) {
    return response.json();
  } else {
    let errorMessage = await response.text();
    switch (response.status) {
      case 404:
        throw new HTTPError(response.status, "File not found");
      case 500:
        throw new HTTPError(response.status, "Server error");
      default:
        throw new HTTPError(response.status, errorMessage || "Unknown error");
    }
  }
};

export default apiRequest;
