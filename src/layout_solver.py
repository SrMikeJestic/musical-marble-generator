"""Trajetórias analíticas no plano XZ, dirigidas pelos instantes das notas.

Cada timestamp, em segundos absolutos, representa um contato com uma plataforma.
As plataformas ficam na mesma altura; seu espaçamento depende do intervalo entre
notas. Para um salto de duração T e tempo local s:

    x(s) = x_inicial + velocidade_x * s
    z(s) = altura_plataforma + (g / 2) * s * (T - s)

Logo, z(0) = z(T) = altura_plataforma, a aceleração vertical é -g e a
velocidade vertical inicial é g*T/2. Cada contato inicia um novo arco, com um
impulso ideal que ajusta essa velocidade à próxima nota. Não há simulação física,
integração numérica, dependências externas ou arredondamento para frames.

Exemplo de uso a partir da raiz do repositório::

    from src.layout_solver import solve_layout

    arcs = solve_layout([0.0, 0.5, 1.25])
    x, z = arcs[0].position_at(0.25)

As coordenadas descrevem o ponto de contato de uma partícula. O raio da esfera
e a geometria das plataformas serão tratados pela futura integração visual.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .config import GRAVITY, HORIZONTAL_SPEED, PLATFORM_Z


@dataclass(frozen=True)
class TrajectorySegment:
    """Um arco entre dois contatos, avaliado por tempo absoluto em segundos."""

    start_time: float
    end_time: float
    start_x: float
    end_x: float
    platform_z: float
    gravity: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def position_at(self, timestamp: float) -> tuple[float, float]:
        """Retorna (X, Z); rejeita instantes fora do intervalo deste arco."""
        if not isfinite(timestamp) or not self.start_time <= timestamp <= self.end_time:
            raise ValueError("O instante deve pertencer ao intervalo do arco.")
        # Preserva exatamente os contatos compartilhados entre arcos consecutivos.
        if timestamp == self.start_time:
            return self.start_x, self.platform_z
        if timestamp == self.end_time:
            return self.end_x, self.platform_z

        elapsed = timestamp - self.start_time
        progress = elapsed / self.duration
        x = self.start_x + (self.end_x - self.start_x) * progress
        z = self.platform_z + 0.5 * self.gravity * elapsed * (self.duration - elapsed)
        return x, z


def solve_layout(
    timestamps: Iterable[float],
    *,
    horizontal_speed: float = HORIZONTAL_SPEED,
    gravity: float = GRAVITY,
    start_x: float = 0.0,
    platform_z: float = PLATFORM_Z,
) -> list[TrajectorySegment]:
    """Gera N-1 arcos e determina as N posições de contato a partir das notas.

    Exige pelo menos dois timestamps finitos, não negativos e estritamente
    crescentes. Notas simultâneas precisam ser agrupadas antes desta função:
    uma única bolinha não percorre duas plataformas no mesmo instante.
    A lista não é ordenada nem quantizada automaticamente.

    Velocidade horizontal e gravidade devem ser finitas e positivas. A primeira
    plataforma fica em (start_x, platform_z); cada arco expõe suas posições
    inicial e final para a construção posterior das plataformas.
    """
    times = list(timestamps)
    if len(times) < 2:
        raise ValueError("Forneça pelo menos dois timestamps.")
    if any(not isfinite(t) or t < 0 for t in times):
        raise ValueError("Os timestamps devem ser finitos e não negativos.")
    if any(end <= start for start, end in zip(times, times[1:])):
        raise ValueError("Os timestamps devem ser estritamente crescentes.")
    if not isfinite(horizontal_speed) or horizontal_speed <= 0:
        raise ValueError("A velocidade horizontal deve ser finita e positiva.")
    if not isfinite(gravity) or gravity <= 0:
        raise ValueError("A gravidade deve ser finita e positiva.")
    if not isfinite(start_x) or not isfinite(platform_z):
        raise ValueError("As coordenadas iniciais devem ser finitas.")

    positions_x = [start_x + horizontal_speed * (t - times[0]) for t in times]
    return [
        TrajectorySegment(
            start_time=start,
            end_time=end,
            start_x=positions_x[index],
            end_x=positions_x[index + 1],
            platform_z=platform_z,
            gravity=gravity,
        )
        for index, (start, end) in enumerate(zip(times, times[1:]))
    ]
