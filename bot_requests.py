from pydantic import BaseModel
from typing import List

class JoinGameRequest(BaseModel):
    player_name: str

class AddBotRequest(BaseModel):
    bot_name: str

class BotPlayer(BaseModel):
    name: str
    algorithm: str


class CreateConfiguredGameRequest(BaseModel):
    human_players: int
    bot_players: List[BotPlayer]

class CreateBotGameRequest(BaseModel):
    bot_players: int

class BotConfig(BaseModel):
    bot_name: str
