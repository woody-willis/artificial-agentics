import "dotenv/config";

import {
    DevelopmentTeamManager,
    DevelopmentTeamManagerInvocationTask,
} from "./teams/development/team-manager";

(async () => {
    const teamManager = await new DevelopmentTeamManager().init();
    console.log("Team manager initialized");

    console.log("Invoking add feature task");
    await teamManager
        .invoke({
            task: DevelopmentTeamManagerInvocationTask.AddFeature,
            data: {
                description:
                    "Modify the existing agents so that they can handle large prompts. They are hitting a 32k token limit at the moment and need to be able to handle larger prompts. The agents are stored in src/teams/development and the tools are in src/tools.",
            },
        })
        .catch((error) => {
            console.error("Error invoking add feature task:", error);
        });
    console.log("Add feature task complete");

    await teamManager.dispose();
})();
