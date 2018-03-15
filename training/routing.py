from channels.routing import route

channel_routing = [
    route('websocket.receive', 'echo.consumers.ws_receive'),
]