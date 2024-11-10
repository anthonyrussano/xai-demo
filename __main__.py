import logging
import json
from typing import List, Dict, Optional, Callable

import hvac
import requests

# Assuming 'vars' module provides VAULT_ADDR, ROLE_ID, and SECRET_ID
import vars

def setup_logging() -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler('xai_chat.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler for ERROR level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

logger = setup_logging()

class VaultClient:
    """Singleton class to manage the Vault client."""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VaultClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Vault client."""
        try:
            VAULT_ADDR = vars.VAULT_ADDR
            ROLE_ID = vars.ROLE_ID
            SECRET_ID = vars.SECRET_ID

            logger.debug(f"Connecting to Vault at {VAULT_ADDR}")
            self._client = hvac.Client(url=VAULT_ADDR)

            # Authenticate using AppRole
            self._client.auth.approle.login(
                role_id=ROLE_ID,
                secret_id=SECRET_ID
            )

            logger.debug("Successfully authenticated with Vault")
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {str(e)}")
            raise

    def get_secret(self, path: str, key: Optional[str] = None) -> Optional[str]:
        """Retrieve secrets from Vault.

        Args:
            path (str): The path to the secret.
            key (Optional[str]): The specific key within the secret data.

        Returns:
            Optional[str]: The secret value or None if not found.
        """
        try:
            # For KV version 2
            secret = self._client.secrets.kv.v2.read_secret_version(
                mount_point='kv',
                path=path
            )

            data = secret.get('data', {}).get('data', {})
            if not data:
                raise ValueError("Secret data not found in expected format")

            # Return specific key if provided, otherwise return all data
            if key:
                return data.get(key)
            return data

        except Exception as e:
            logger.error(f"Error retrieving secret: {str(e)}")
            raise

class XAIChat:
    """Client for interacting with the xAI API."""

    def __init__(self, api_key: str):
        """Initialize XAI Chat client with API key.

        Args:
            api_key (str): The API key for authentication.

        Raises:
            ValueError: If the API key is empty.
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        logger.debug(f"Initialized XAIChat with base URL: {self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        on_data: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Send a chat request to the xAI API.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            on_data (Callable[[str], None], optional): Function to call with each chunk of data.

        Returns:
            Optional[str]: The assistant's reply or None if an error occurs.
        """
        try:
            data = {
                "messages": messages,
                "model": "grok-beta",
                "stream": stream,
                "temperature": temperature
            }

            logger.debug(f"Making request to {self.base_url}/chat/completions")
            logger.debug(f"Request data: {json.dumps(data, indent=2)}")

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                stream=stream
            )

            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")

            response.raise_for_status()

            if stream:
                return self._handle_stream(response, on_data=on_data)
            else:
                return self._handle_regular_response(response)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            if e.response is not None:
                logger.error(f"Response text: {e.response.text}")
            return None

    def _handle_regular_response(self, response: requests.Response) -> Optional[str]:
        """Handle a regular (non-streamed) response.

        Args:
            response (requests.Response): The HTTP response object.

        Returns:
            Optional[str]: The assistant's reply or None if parsing fails.
        """
        try:
            result = response.json()
            logger.debug(f"Received response: {json.dumps(result, indent=2)}")
            return result['choices'][0]['message']['content']
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None

    def _handle_stream(
        self,
        response: requests.Response,
        on_data: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Handle a streamed response.

        Args:
            response (requests.Response): The HTTP response object.
            on_data (Callable[[str], None], optional): Function to call with each chunk of data.

        Returns:
            Optional[str]: The assistant's full reply or None if an error occurs.
        """
        full_response = []
        try:
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    logger.debug(f"Received line: {line_text}")

                    if line_text.startswith('data: '):
                        if line_text.strip() == 'data: [DONE]':
                            break

                        json_str = line_text[6:]  # Remove 'data: ' prefix
                        data = json.loads(json_str)

                        # Check for delta or message content
                        content = None
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            message = data['choices'][0].get('message', {})
                            content = delta.get('content') or message.get('content')

                        if content:
                            full_response.append(content)
                            if on_data:
                                on_data(content)

            return ''.join(full_response)

        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            return None

def main():
    try:
        # Initialize Vault client and get API key
        vault_client = VaultClient()
        api_key = vault_client.get_secret("xai", "personal")
        if not api_key:
            raise ValueError("Failed to retrieve API key from Vault")

        # Initialize the chat client
        chat_client = XAIChat(api_key)

    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        print(f"Error: {e}")
        return

    # Set up the system message
    messages = [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        }
    ]

    print("xAI Chat Interface (Type 'quit' to exit)")
    print("-" * 50)

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Add user message to the conversation
        messages.append({"role": "user", "content": user_input})

        # Prepare to capture the assistant's reply
        assistant_reply = ''

        # Define the callback function for streaming data
        def on_data(content: str):
            nonlocal assistant_reply
            print(content, end='', flush=True)
            assistant_reply += content

        # Get response
        print("\nGrok:", end=" ", flush=True)
        response = chat_client.chat(
            messages,
            temperature=0.7,
            stream=True,
            on_data=on_data
        )

        if response:
            # Ensure the cursor moves to the next line after the response
            print()
            # Add assistant's response to the conversation history
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            print("\nError: Failed to get response from API")
            logger.error("No response received from API")

if __name__ == "__main__":
    main()
