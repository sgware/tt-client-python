"""
This library provides the abstract classes `tt.Client` and `tt.Factory` for
implementing [Tandem Tales](https://tandemtales.net) agents in Python.

`Client` handles connecting to the [server](https://github.com/sgware/tt-server)
and joining a session. A minimal subclass of `Client` only needs to implement
the `on_choice(self, status)` method to choose what the agent will do on its
turn.

`Factory` continuously creates new clients to connect to the server and wait for
partners. A minimal subclass of `Factory` only needs to implement the
`create(self)` method to return a new Client instance.
"""
from abc import ABC, abstractmethod
import queue
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import socket
import ssl
import json
import requests

VERSION = "0.10.0"
"""The client's version of the Tandem Tales protocol."""

DEFAULT_PORT = 9005
"""The default port on which to connect to the server."""

PLAYER = 'PLAYER'
"""The string used by the Tandem Tales server for the player role."""

GM = 'GAME_MASTER'
"""The string used by the Tandem Tales server for the game master role."""

PROPOSE = 'PROPOSE'
"""The string used by the Tandem Tales server for turn types that propose an action."""

SUCCEED = 'SUCCEED'
"""The string used by the Tandem Tales server for turn types that execute an action."""

FAIL = 'FAIL'
"""The string used by the Tandem Tales server for turn types where an action fails."""

PASS = 'PASS'
"""The string used by the Tandem Tales server for turn types that pass control to the partner."""

ENV_PASSWORD = "password"
"""The name of the environment variable where the agent's password is stored."""

ENV_API_KEY = "apikey"
"""The name of the environment variable where the agent's API key is stored."""

EMBED_URL = 'https://openrouter.ai/api/v1/embeddings'
"""The URL to which large language model text embedding requests will be sent."""

EMBED_MODEL = 'qwen/qwen3-embedding-8b'
"""The model to be used for large language model text embedding requests."""

EMBED_TOKENS_LIMIT = 1000000
"""The maximum number of input tokens that can be used for text embedding requests."""

COMPLETE_URL = 'https://openrouter.ai/api/v1/chat/completions'
"""The URL to which large language model text completion requests will be sent."""

COMPLETE_MODEL = 'deepseek/deepseek-v4-flash-0731'
"""The large language model to be used for text completion requests."""

COMPLETE_IN_TOKENS_LIMIT = 1000000
"""The maximum number of input tokens that can be used for text completion requests."""

COMPLETE_OUT_TOKENS_LIMIT = 1000000
"""The maximum number of output tokens that can be used for text completion requests."""

COMPLETE_SEED = 0
"""The seed used for large language model text completion requests."""

class _ClientInput(threading.Thread):
    """
    A utility that listens for input on the client's socket on a separate
    thread, parses JSON messages, and stores them in a queue.
    """
    
    def __init__(self, socket):
        super().__init__()
        self.input = socket.makefile('r')
        self.received = queue.Queue()
        self.stopped = False
    
    def run(self):
        """
        Listen to the client's socket for input and put any messages received
        into the queue.
        """
        while True:
            # Read next line of input.
            line = None
            try:
                line = self.input.readline()
            except:
                line = ''
            # If the socket has closed, add None to the queue.
            if line == '':
                self.received.put(None)
                break
            else:
                # Parse the message as JSON.
                try:
                    message = json.loads(line)
                    if message['type'] == 'Stop':
                        self.stopped = True
                    self.received.put(message)
                # If an error occurs while parsing, add the exception to the queue.
                except Exception as exception:
                    self.received.put(exception)
                    break
    
    def receive(self, type=None):
        """
        Returns the next message waiting in the queue or blocks until a message
        becomes available. Once the socket has been closed, this method will
        return None. If `type` is provided, this method will raise an exception
        unless the next message is of the expected type. If an exception occurred
        on the thread that was receiving and parsing messages, this method will
        raise that exception so that it can be handled on the client's main
        thread.
        """
        message = self.received.get()
        if message == None:
            return None
        elif isinstance(message, BaseException):
            raise message
        elif type == None or message['type'] == type:
            return message
        else:
            excp = f"Expected \"{type}\" message but received \"{message['type']}\" message"
            if message['type'] == 'Error':
                excp += ': \"' + message['message'] + '\"'
            else:
                excp += '.'
            raise ValueError(excp)

class _ClientOutput:
    """
    A utility that allows the client to send messages via JSON.
    """
    
    def __init__(self, socket):
        self.socket = socket
    
    def send(self, message):
        """
        Convert the message object to JSON and send it via the client socket's
        output followed by a new line.
        """
        message = json.dumps(message) + "\n"
        self.socket.sendall(message.encode())

class Client(ABC, threading.Thread):
    """
    A Tandem Tales client connects to a server to participate in a paired
    storytelling session. The abstract method `on_choice` must be implemented
    to return the index of the desired choice when it is this client's turn.
    A client can be run as thread or directly by calling the `run` method.
    
    Attributes:
        factory (Factory): The factory object that created this client.
        external (ExternalAPI): The external API used for text embedding and
            completion.
        join (dict[str, Any]): This client's join message.
        url (str): The URL of the server to connect to.
        port (int): The port number to connect on.
        world (dict[str, Any]): Will be set to a world object once the session
            starts.
        role (str): The client's role in the story. Updated after session
            starts.
        status (dict[str, Any]): The most recent story world status received
            from the server.
        choices (list[dict[str, Any]]): The list of choices currently available
            to make.
        stop (dict[str, Any]): The stop message received from the server.
        session (str): The session ID received from the server after the session.
    
    Arguments:
        name (str): The name this client will use on the server.
        password (str): The password this client will use when connecting to the
            server. If None, will be read from the environment variables.
        world (str): The name of the story world this client wants to play in.
            None means willing to play in any world.
        role (str): Must be PLAYER or GM. All other values are treated as
            None, which means willing to play either role.
        partner (str): The name of the partner this client wants to play with.
            None means willing to play with any partner.
        key (str): The key this client will use to access the external API. If
            None, will be read from the environment variables.
        url (str): The URL of the server this client will connect to.
        port (int): The port number this client will connect to the server on.
    """
    
    def __init__(self, name, password, world, role, partner, key, url='localhost', port=DEFAULT_PORT):
        super().__init__()
        if password is None and ENV_PASSWORD in os.environ:
            password = os.environ[ENV_PASSWORD]
        if key is None and ENV_API_KEY in os.environ:
            key = os.environ[ENV_API_KEY]
        self.factory = None
        self.external = ExternalAPI(key)
        self.join_message = {
            "type": "Join",
            "name": name,
            "password": password,
            "world": world,
            "role": role,
            "partner": partner
        }
        self.url = url
        self.port = port
        self.socket = None # Will be set when the agent connects.
        self.world = None # Will be set to the world object after session starts.
        self.role = role # Will be updated after session starts.
        self.status = None
        self.choices = None
        self.stop = None
        self.session = None
    
    def __str__(self):
        str = f"[Client: name={self.join_message['name']}"
        if 'password' in self.join_message:
            str += '; password="***"'
        if 'world' in self.join_message:
            str += f"; world=\"{self.join_message['world']}\""
        if 'role' in self.join_message:
            str += f"; role=\"{self.join_message['role']}\""
        if 'partner' in self.join_message:
            str += f"; partner=\"{self.join_message['partner']}\""
        return str + ']'
    
    def run(self):
        """
        Connects to the server, joins, waits for a session to start, sends the
        choices made by this client, and returns the session ID of the completed
        session. As this method runs, it calls other methods to notify the
        client of important events. Some of these methods are guaranteed to be
        called even if an exception is raised. The list of those methods is:
        
        1. `connect` is called to establish the socket. If this method raises
           an exception, it is raised immediately and no other methods run.
        2. `on_connect` is called after receiving the connect message from the
           server. This is the last chance to make changes to the join request.
        3. `on_start` is called after the session starts.
        4. `on_update` is called each time the client receives an update to the
           story world status from the server.
        5. `on_choice` is called when the client receives an update that
           requires it to make a choice of what action to take next.
        6. `on_end` is called if the story reaches a pre-defined ending.
        7. `on_close` is called when the client stops its session normally,
           like reaching the end of the story, closing, or disconnecting. If
           the client stopped because it was interrupted or because of an
           exception, this method is not called.
        8. `on_stop` will always be called if the client's session started.
        9. `on_disconnect` will always be called if the client connected.
        """
        # Warn if the password and API key are not set.
        if self.join_message['password'] == None and not ENV_PASSWORD in os.environ:
            self.on_warning(f"The environment variable \"{ENV_PASSWORD}\" is not set. This agent will not use a password.")
        if self.external.key == None and not ENV_API_KEY in os.environ:
            self.on_warning(f"The environment variable \"{ENV_API_KEY}\" is not set. This agent will not be able to use the external API.")
        # Connect to the server.
        self.connect()
        # If an exception is raised, save it to raise later.
        uncaught = None
        try:
            # Wait for the connect message.
            connect = self.receive('Connect')
            if connect != None:
                if connect['version'] != VERSION:
                    self.on_warning(f"This client is using version {VERSION} of the communication protocol, but the server is using version {connect['version']}. This may cause miscommunications.")
                self.on_connect(connect)
                self.send(self.join_message)
            # Wait for the start message.
            start = self.receive('Start')
            if start != None:
                self.world = start['world']
                self.role = start['role']
                if self.factory != None:
                    self.factory._on_start(self)
            # Receive and process messages from the server.
            while self.world != None:
                message = self.receive()
                # Stop if disconnected.
                if message == None:
                    break
                # Handle update messages.   
                elif message['type'] == "Update":
                    # If this is the first update, notify the client the session has started.
                    if self.status == None:
                        self.status = message['status']
                        self.choices = message['status']['choices']
                        self.on_start(self.world, self.role, self.status['state'])
                    # If this is not the first update, update the current status and choices.
                    else:
                        self.status = message['status']
                        self.choices = message['status']['choices']
                        self.on_update(self.status)
                    # If it is this client's turn, make a choice.
                    if len(self.choices) > 0:
                        index = self.on_choice(self.status)
                        self.choices = json.loads('[]')
                        choice = {
                            "type": "Choice",
                            "index": index
                        }
                        if not self.input.stopped:
                            self.send(choice)
                    # If the story has ended, notify this client.
                    elif self.status.get('ending') is not None:
                        self.on_end(self.status['ending'])
                # Immediately acknowledge stop messages.
                elif message['type'] == "Stop":
                    self.stop = message
                    stop = {
                        "type": "Stop",
                        "role": self.role
                    }
                    self.send(stop)
                    self.on_stop(self.stop['message'])
                # Get session ID from end message.
                elif message['type'] == "End":
                    self.session = message['session']
                    break
                # Handle errors reported by the server.
                elif message['type'] == "Error":
                    self.on_error(message['message'])
                # Handle unrecognized message types.
                else:
                    m = str(message)
                    if len(m) > 40:
                        m = m[:40] + '...'
                    self.on_error(f"The message \"{m}\" is not recognized.")
        # Register an uncaught exception.
        except Exception as exception:
            uncaught = exception
        # If the client is interrupted...
        except KeyboardInterrupt as interrupt:
            if uncaught == None:
                uncaught = Exception("Client was interrupted.")
        # If this client ended normally, close.
        if uncaught == None:
            try:
                self.on_close()
            except Exception as exception:
                uncaught = exception
        # If the session started but has not stopped, stop it.
        if self.world != None and self.stop == None:
            try:
                self.on_stop(None)
            except Exception as exception:
                if uncaught == None:
                    uncaught = exception
        # Close the socket.
        self.close()
        # Let this client know it has disconnected.
        try:
            self.on_disconnect()
        except Exception as exception:
            if uncaught == None:
                uncaught = exception
        # If there were no problems, return the session ID.
        if uncaught == None:
            return self.session
        else:
            raise uncaught
    
    def connect(self):
        """
        Creates a secure connection to the server and creates the client input
        and output utilities.
        """
        context = ssl.create_default_context()
        s = socket.create_connection((self.url, self.port))
        self.socket = context.wrap_socket(s, server_hostname=self.url)
        self.socket.do_handshake()
        self.input = _ClientInput(self.socket)
        self.output = _ClientOutput(self.socket)
        self.input.start()
    
    def receive(self, type=None):
        """
        Receives a message from the server as an object parsed from JSON,
        waiting if one is not available. None means the socket has been closed.
        If an exception was raised on the client listener thread, it is raised
        when this method is called instead of on that thread.
        """
        r = self.input.receive(type)
        return r
    
    def send(self, message):
        """
        Converts an object to JSON and send it to the server.
        """
        self.output.send(message)
    
    def close(self):
        """
        Closes this client's socket.
        """
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except:
                pass
    
    def on_connect(self, connect):
        """
        Called after receiving the connect message from the server.
        """
        pass
    
    def on_start(self, world, role, state):
        """
        Called after the session starts.
        
        Arguments:
            world (dict[str, Any]): An object detailing the entities, variables,
                actions and endings available in this session's story world.
            role (str): The role this client is playing, which will be either
                PLAYER or GM.
            state (dict[str, Any]): An object giving the current value of all
                variables when the story starts.
        """
        pass
    
    def on_update(self, status):
        """
        Called when the server sends a story world status update.
        
        status['history'] is all turns that have happened so far.
        status['state'] gives the current value of all variables.
        status['ending'] gives the story's ending, or None if it is ongoing.
        status['descriptions'] gives descriptions of all visible entities.
        """
        pass
    
    @abstractmethod
    def on_choice(self, status) -> int:
        """
        Called when the server sends a story world status update that requires
        this client to choose what turn will happen next. This method must
        return the index (starting at 0) of the action it wants to take. The
        list of choices can be found in status['choices'].
        
        status['history'] is all turns that have happend so far.
        status['state'] gives the current value of all variables.
        status['ending'] gives the story's ending, or None if it is ongoing.
        status['descriptions'] gives descriptions of all visible entities.
        status['choices'] gives the list of currently available turns.
        """
        pass
    
    def on_end(self, ending):
        """
        Called when the story reaches one of its pre-defined endings.
        
        The `ending` object has a description of how the story ended.
        """
        pass
    
    def on_stop(self, message):
        """
        Always called at the end of the client's life if the session started.
        
        The `message` is a string explaining why the client stopped, and it may
        be None.
        """
        pass
    
    def on_close(self):
        """
        Called if the client stopped normally or because it was closed or
        disconnected. Not called if the client stopped because an exception was
        raised.
        """
        pass
    
    def on_disconnect(self):
        """
        Always called at the end of the client's life if it connected.
        """
        pass
    
    def on_warning(self, message):
        """
        Called if a non-fatal problem occurs with a string explanation.
        """
        print(f"Warning: {message}")
    
    def on_error(self, message):
        """
        Called if a fatal problem occurs with a string explanation.
        """
        raise RuntimeError(f"Error: {message}")

class ExternalAPI:
    """
    Represents the external API used for text embedding and completion.
    
    Attributes:
        key (str): The API key used to make requests.
        embed_tokens (int): The number of input tokens used so far by the text
            embedding model.
        complete_in_tokens (int): The number of input tokens used so far by the
            text completion model.
        complete_out_tokens (int): The number of output tokens used so far by
            the text completion model.
    
    Arguments:
        key (str): The API key used to make requests.
    """
    
    def __init__(self, key):
        self.key = key
        self.embed_tokens = 0
        self.complete_in_tokens = 0
        self.complete_out_tokens = 0
    
    def _check_key(self):
        """
        Raises an exception if the API key is not set.
        """
        if self.key == None:
            raise ValueError('This agent does not have an API key. It cannot use the external API.')
    
    def _check_in_limit(self, used, limit):
        """
        Raise an exception if the number of input tokens used exceeds the limit.
        """
        if used >= limit:
            raise ValueError(f"The input token limit {limit} has been exceeded.")
    
    def _check_out_limit(self, used, limit):
        """
        Raise an exception if the number of output tokens used exceeds the limit.
        """
        if used >= limit:
            raise ValueError(f"The output token limit {limit} has been exceeded.")
    
    def _check_nominal(self, name, value, values):
        """
        Raise an exception if the value given is not a string from a pre-defined
        list.
        """
        if isinstance(value, str) and value in values:
            pass
        else:
            raise ValueError(f"{name} must be one of {values}.")
    
    def _check_int(self, name, value, min=None, max=None):
        """
        Raise an exception if the given value is not an integer in the given
        range.
        """
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        elif min != None and value < min:
            raise ValueError(f"{name} must be >= {min}.")
        elif max != None and value > max:
            raise ValueError(f"{name} must be <= {max}.")

    def _check_float(self, name, value, min=None, max=None):
        """
        Raise an exception if the given value is not a float in the given range.
        """
        if not isinstance(value, float):
            raise ValueError(f"{name} must be a float.")
        elif min != None and value < min:
            raise ValueError(f"{name} must be >= {min}.")
        elif max != None and value > max:
            raise ValueError(f"{name} must be <= {max}.")
    
    def embed(self, text, dim=1024):
        """
        Queries the external text embedding API and returns its response as an
        array of floats. Each time this method is called, the number of tokens
        used is added to `embed_tokens`. If the number of tokens used exceeds
        `EMBED_TOKENS_LIMIT`, this method will raise an exception.
        
        Arguments:
            text (str): The text which will be embedded as an array of floats.
            dim (int): The number of dimensions in the returned array.
        """
        # Check if this request is possible.
        self._check_key()
        self._check_in_limit(self.embed_tokens, EMBED_TOKENS_LIMIT)
        # Construct the data object.
        data = {
            'model': EMBED_MODEL,
            'input_type': 'string',
            'input': text,
            'encoding_format': 'float'
        }
        if dim != None:
            self._check_int('Dimensions', dim)
            data['dimensions'] = dim
        # Send the request.
        response = requests.post(
            url=EMBED_URL,
            headers={
                'Authorization': f"Bearer {self.key}",
                'Content-Type': 'application/json',
            },
            data=json.dumps(data)
        )
        # Process the response.
        response.raise_for_status()
        response_data = json.loads(response.text)
        self.embed_tokens += response_data['usage']['total_tokens']
        return response_data['data'][0]['embedding']
    
    def complete(self, messages, maxt=None, reasoning='low', temp=0.0, topk=None, topp=None, minp=None):
        """
        Queries the external text completion API and returns its response as a
            string. Each time this method is called, the number of tokens used
            in the prompt is added to `complete_in_tokens` and the number of
            tokens used in reasoning and completion is added to
            `complete_out_tokens`. If the number of tokens used exceeds
            `COMPLETE_IN_TOKENS_LIMIT` or `COMPLETE_OUT_TOKENS_LIMIT`
            respectively, this method will raise an exception.
        
        Arguments:
            messages (str): This can be either a single string or an array of
                strings representing a conversation. If it is a single string,
                the model will use its default system prompt and treat the
                string as a message from the user. If it is an array of strings,
                it must have an even numbered length. The first string will be
                used as the system prompt. Every odd numbered message will be
                treated as a message from the user. Every even numbered message
                (after the first) will be treated as a response from the
                assistant.
            maxt (int): The maximum number of tokens the model may use to
                generate its response.
            reasoning (str): The level of reasoning used by the large language
                model during text completion. Values include `max`, `high`,
                `low`, and `none`.
            temp (float): Influences the randomness of the response. A value of
                0.0 will give deterministic responses. The max value is 2.0.
            topk (int): When choosing a next token to add to the output, the
                model may only choose from the best k tokens.
            topp (float): When choosing a next token to add to the output, the
                model may only choose from tokens whose probability add to p.
            minp (float): When choosing a next token to add to the output, the
                model may only choose from tokens whose probability is p times
                the probability of the most likely token.
        """
        # Check if this request is possible.
        self._check_key()
        self._check_in_limit(self.complete_in_tokens, COMPLETE_IN_TOKENS_LIMIT)
        self._check_out_limit(self.complete_out_tokens, COMPLETE_OUT_TOKENS_LIMIT)
        # Construct the conversation.
        msg = []
        # If messages is a single string, use no system prompt.
        if isinstance(messages, str):
            msg.append({ 'role': 'user', 'content': messages })
        # Otherwise, assume messages is an array of strings.
        else:
            if(len(messages) % 2 == 1):
                raise ValueError('Conversation must have an even number of messages.')
            msg.append({ 'role': 'system', 'content': messages[0] })
            for i, m in enumerate(messages[1:], start=1):
                if i % 2 == 1:
                    msg.append({ 'role': 'user', 'content': m })
                else:
                    msg.append({ 'role': 'assistant', 'content': m })
        # Construct the data object.
        data = {
            'model': COMPLETE_MODEL,
            'messages': msg,
            'seed': COMPLETE_SEED
        }
        if maxt != None:
            self._check_int('Max output tokens', maxt, 1)
            data['max_completion_tokens'] = maxt
        if reasoning != None:
            self._check_nominal('Reasoning level', reasoning, {'max', 'high', 'low', 'none'})
            data['reasoning_effort'] = reasoning
        if temp != None:
            self._check_float('Temperature', temp, 0.0, 2.0)
            data['temperature'] = temp
        if topk != None:
            self._check_int('Top k', topk)
            data['top_k'] = topk
        if topp != None:
            self._check_float('Top p', topp, 0.0, 1.0)
            data['top_p'] = topp
        if minp != None:
            self._check_float('Min p', minp, 0.0, 1.0)
            data['min_p'] = minp
        # Send the request.
        response = requests.post(
            url=COMPLETE_URL,
            headers={
                'Authorization': f"Bearer {self.key}",
                'Content-Type': 'application/json',
            },
            data=json.dumps(data)
        )
        # Process the response.
        response.raise_for_status()
        response_data = json.loads(response.text)
        self.complete_in_tokens += response_data['usage']['prompt_tokens']
        self.complete_out_tokens += response_data['usage']['completion_tokens']
        return response_data['choices'][0]['message']['content']

class _Operation(ABC):
    """
    An operation performed by a Factory that will report any exceptions raised
    to that factory and then close the factory.
    """
    
    def __init__(self, factory):
        self.factory = factory
    
    def __call__(self):
        try:
            self.run()
        except Exception as exception:
            if self.factory.uncaught == None:
                self.factory.uncaught = exception
            self.factory.close()
    
    @abstractmethod
    def run(self):
        pass

class _StartFirstClient(_Operation):
    """ Starts the first client. """
    
    def __init__(self, factory):
        super().__init__(factory)
    
    def run(self):
        self.factory._create()

class _RunClient(threading.Thread, _Operation):
    """ Starts a client and ensures it is stopped. """
    
    def __init__(self, factory, client):
        threading.Thread.__init__(self)
        _Operation.__init__(self, factory)
        self.client = client
    
    def run(self):
        try:
            self.client.run()
        finally:
            self.factory._on_stop(self.client)

class _StartClient(_Operation):
    """ Move a client from waiting to running and start a new one if allowed. """
    
    def __init__(self, factory, client):
        super().__init__(factory)
        self.client = client
    
    def run(self):
        self.factory.waiting.remove(self.client)
        self.factory.running.append(self.client)
        if self.factory.max <= 0 or len(self.factory.waiting) + len(self.factory.running) < self.factory.max:
            self.factory._create()

class _StopClient(_Operation):
    """ Remove a client from a factory and start a new one if allowed. """
    
    def __init__(self, factory, client):
        super().__init__(factory)
        self.client = client
    
    def run(self):
        if self.client in self.factory.running:
            self.factory.running.remove(self.client)
            if len(self.factory.waiting) == 0:
                self.factory._create()
        else:
            self.factory.waiting.remove(self.client)

class _CloseFactory(_Operation):
    """ Begin a factory's shut down process. """
    
    def __init__(self, factory):
        super().__init__(factory)
    
    def run(self):
        self.factory.closed = True

class Factory(ABC):
    """
    A factory continuously creates clients so that when a client starts its
    session a new one will take its place waiting on the server. The abstract
    method `create` must be implemented; it should return a new client object.
    
    Attributes:
        max (int): The maximum number of clients that may be running at a time
            or 0 for no limit.
        queue (queue.Queue): A queue of operations waiting to run on the
            factory's thread.
        waiting (list[Client]): The clients currently connected to the server
            and waiting.
        running (list[Client]): The clients currently in sessions.
        uncaught (Exception): An exception raised on another thread waiting to
            be raised on the factory's thread.
        closed (boolean): If the factory has been closed.
    
    Arguments:
        max (int): The maximum number of clients that may be running at a time
            or 0 for no limit.
    """
    
    def __init__(self, max=0):
        self.max = max
        self.queue = queue.Queue()
        self.waiting = []
        self.running = []
        self.executor = ThreadPoolExecutor()
        self.uncaught = None
        self.closed = False
    
    def __str__(self):
        tos = '[Client Factory: '
        if self.max > 0:
            tos += str(len(self.waiting) + len(self.running)) + '/' + str(self.max) + ' clients; '
        tos += f"{len(self.waiting)} waiting; {len(self.running)} running]"
        return tos
    
    def run(self):
        """
        Starts the factory, starts the first client, and creates new clients as
        allowed when clients start or end their sessions. Each client starts on
        a different thread, but if they raise exceptions, they will be handled
        on this thread. Certain things are guaranteed to happen in order:
        
        1. `on_start` is called when this method starts. If it raises an
           exception, no other methods are called.
        2. The first client is created via `create` and starts.
        3. Each time a client starts its session, a new client is created to
           take its place on the server, unless this would violate the
           factory's limit on the max number of clients.
        4. `on_close` runs when the factory is closed via the `close` method or
           if a client raises an exception. `on_close` is not called if the
           factory was interrupted. `on_close` always runs on the thread that
           called this method, even if `close` was called by a different thread.
        5. Before shutting down, the factory closes all clients that have not
           started their sessions and waits for all clients that have started
           their sessions to stop.
        6. `on_stop` is always called after all clients have stopped.
        7. If an exception was raised at any point, it is thrown after
           `on_stop`. If everything shut down normally, this method returns
           None.
        """
        # Let this factory know it has started.
        self.on_start()
        try:
            # Start the first client.
            self.queue.put(_StartFirstClient(self))
            # Run commands until closed.
            while not self.closed:
                operation = self.queue.get()
                operation()
            # Let this factory know it has closed.
            self.on_close()
        except Exception as exception:
            if self.uncaught == None:
                self.uncaught = exception
        except KeyboardInterrupt as interrupt:
            if self.uncaught == None:
                self.uncaught = Exception("Factory was interrupted.")
        # Close waiting clients.
        for client in self.waiting:
            client.close()
        # Wait for all clients to stop.
        self.executor.shutdown(wait=True)
        # Let this factory know it has stopped.
        try:
            self.on_stop()
        except Exception as exception:
            if self.uncaught == None:
                self.uncaught = exception
        # If there were no problems, return None.
        if self.uncaught == None:
            return None
        else:
            raise self.uncaught
    
    def on_start(self):
        """
        Called when this factory starts. This is a good place to set up assets
        before any clients start.
        """
        pass
    
    def _create(self):
        client = self.create()
        client.factory = self
        self.waiting.append(client)
        self.executor.submit(_RunClient(self, client))
    
    @abstractmethod
    def create(self) -> Client:
        """
        Create and return a new Tandem Tales client object. This method always
        runs on the same thread that called `run`.
        """
        pass
    
    def _on_start(self, client):
        self.queue.put(_StartClient(self, client))
    
    def _on_stop(self, client):
        self.queue.put(_StopClient(self, client))
    
    def close(self):
        """
        Stop this factory from creating any new clients and begin the shutdown
        process. This method can be safely called from any thread.
        """
        self.queue.put(_CloseFactory(self))
    
    def on_close(self):
        """
        Runs after this factory has been closed either by the `close` method or
        because an exception was raised by the factory or one of its clients.
        This method always runs on the same thread that called `run` even if
        `close` was called on a different thread.
        """
        pass
    
    def on_stop(self):
        """
        Called after all clients have stopped and this factory is about to stop.
        This is a good place to clean up.
        """
        pass
