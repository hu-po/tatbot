# $TATBOT_ROOT/tatbot/rr.py

import rerun as rr
import rerun.blueprint as rrb

def make_blueprint() -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Spatial2DView(),

        )
    )
    return blueprint

def main():
    rr.init("tabtot", spawn=True)
    rr.send_blueprint(make_blueprint())

if __name__ == "__main__":
    main()