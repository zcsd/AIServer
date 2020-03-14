#!/usr/bin/python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np
from datetime import datetime
from dqn import DQNAgent

PORT = 8000

class ResquestHandler(BaseHTTPRequestHandler):
	def __init__(self, request, client_address, server):
		self.diffAgent = DQNAgent(True, 'diffusion')
		self.osmoAgent = DQNAgent(True, 'osmosis')
		BaseHTTPRequestHandler.__init__(self, request, client_address, server);
		
	def do_GET(self):
		content_len = int(self.headers.get('Content-Length'))
		if (content_len > 0):
			body = self.rfile.read(content_len) # bytes
			input_state = body.decode(encoding="utf-8", errors="strict") # string
			json_input_state = json.loads(input_state) # json
			print("State Received: ", json_input_state)
			
			username = json_input_state['username']
			sequenceID = json_input_state['sequenceID']
			stage = json_input_state['stage']
			itemsState = json_input_state['itemsState']
			####################Do Predication####################
			lis_state = []
			for state in itemsState:
				lis_state.append(int(state))
			
			np_state = np.reshape(lis_state,(1,27))
			if stage == 'diffusion':
				print('diff agent is used.')
				result = self.diffAgent.model.predict(np_state)
			elif stage == 'osmosis':
				print('osmo agent is used.')
				result = self.osmoAgent.model.predict(np_state)
			#print("Model predicted result : ", result)
			decision = np.argmax(result)
			#print("Result index : ", decision)
			hint = self.process_decision(decision);
			#####################Predication End####################
			output_hint = {'username': username, 'sequenceID': sequenceID,'stage': stage, 'hint': hint}
			
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(json.dumps(output_hint).encode("utf-8"))
			print("Sent a hint: ", hint)
		else:
			print("Invalid request")
	
	def process_decision(self, choice):
		index = choice % 3
		itemno = choice//3

		if index==0:
			action = "购买"
		elif index ==1:
			action = "使用" 
		else: 
			action = "return"

		decision = action + "物品" + str(itemno)

		return decision
		

if __name__ == '__main__':
	server = HTTPServer(('', PORT), ResquestHandler)
	print("Starting HTTP server, listen at: %s" % PORT)
	server.serve_forever()
